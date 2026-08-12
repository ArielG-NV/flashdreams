# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-window session edges."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

from flashdreams.runtime import (
    StepResult,
    UserInputCapability,
    UserInputEvent,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    AlwaysActiveActivationPolicy,
    InMemorySessionMetricsRecorder,
    ModelInputProvider,
    NativeWindowErrorPolicy,
    NoopTransportService,
    OutputDecision,
    PreparedScenario,
    RealtimeEventInputSource,
    RealtimeSessionDriver,
    ResamplerRealtimeClock,
    RunContext,
    RunModeCapabilities,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
)
from flashdreams.runtime.demo.spec import DemoAdapter, DemoSpec
from flashdreams.runtime.demo.timing import RealtimeEventResampler


class NativeFrameQueue:
    """Bounded queue that drops whole stale chunks."""

    def __init__(self, *, max_chunks: int) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0.")
        self._max_chunks = max_chunks
        self._chunks: deque[deque[object]] = deque()
        self._lock = threading.Lock()
        self.closed = False

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def publish(self, result: StepResult) -> tuple[bool, int]:
        frames = deque[object](result.lazy_rgb_frames(record_cuda_event=True))
        with self._lock:
            if self.closed:
                return True, 0
            dropped = len(self._chunks) >= self._max_chunks
            if dropped:
                self._chunks.popleft()
            if frames:
                self._chunks.append(frames)
            return dropped, sum(len(chunk) for chunk in self._chunks)

    def pop(self) -> object | None:
        with self._lock:
            if not self._chunks:
                return None
            frame = self._chunks[0].popleft()
            if not self._chunks[0]:
                self._chunks.popleft()
            return frame

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._chunks.clear()


class NativeWindowOutputSink:
    """Send generated video frames to a native presentation queue."""

    produces_artifacts = False

    def __init__(self, *, queue: NativeFrameQueue, fps: int) -> None:
        self._queue = queue
        self._fps = fps
        self._open = False

    def open(self, session_info: SessionInfo) -> None:
        del session_info
        self._open = True
        self._queue.clear()

    def begin_generation(self, generation: int) -> None:
        del generation
        self._queue.clear()

    def write(self, result: StepResult) -> OutputDecision:
        if not self._open:
            raise RuntimeError("Cannot write to a closed native-window sink.")
        dropped, queued = self._queue.publish(result)
        return OutputDecision(
            dropped=dropped,
            drop_policy="drop_oldest" if dropped else "none",
            backpressure_s=queued / float(self._fps),
        )

    def close(self) -> Sequence[Any]:
        self._open = False
        self._queue.close()
        return ()


_NATIVE_INPUT_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
        UserInputCapability(
            event_type="text_event", payload_fields=frozenset({"event_id", "state"})
        ),
    )
)


class NativeWindowInputSource(RealtimeEventInputSource):
    """Thread-safe source for native camera keys and text events."""

    def __init__(self, *, fps: int) -> None:
        self._lock = threading.RLock()
        super().__init__(
            resampler=RealtimeEventResampler(fps=fps),
            user_input_schema=_NATIVE_INPUT_SCHEMA,
        )

    def record_key(self, *, event: str, key: str, timestamp_s: float) -> None:
        event_type = {"keydown": "key_down", "keyup": "key_up"}.get(event.lower())
        if event_type is None or not key.strip():
            raise ValueError("Native key events require keydown/keyup and a key.")
        self.record_user_event(
            UserInputEvent(
                timestamp_s=timestamp_s,
                event_type=event_type,
                payload={"key": key.strip()},
                source="native-window",
            )
        )

    def record_user_event(self, event: UserInputEvent) -> None:
        with self._lock:
            super().record_user_event(event)

    def _events_for_window(self, start_s: float, end_s: float):
        with self._lock:
            return super()._events_for_window(start_s, end_s)

    def _prune_events(self, *, before_s: float) -> None:
        with self._lock:
            super()._prune_events(before_s=before_s)

    def reset(self, *, start_v: float) -> None:
        with self._lock:
            super().reset(start_v=start_v)


class NativeWindowRunMode:
    """Realtime run mode for a local window."""

    name = "native-window"
    capabilities = RunModeCapabilities(
        realtime=True,
        supports_backpressure=True,
        supports_interactive_events=True,
    )

    def __init__(self, *, input_source, output_sink, transport) -> None:
        self.input = input_source
        self.output = output_sink
        self.transport = transport

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        del adapter
        if spec.output.mode != "native-window":
            raise ValueError("NativeWindowRunMode requires native-window output.")

    def validate_session(
        self, *, spec, scenario: PreparedScenario, adapter, provider: ModelInputProvider
    ) -> None:
        del spec, scenario, adapter
        if not provider.capabilities.supports_realtime_clock:
            raise ValueError("Native-window providers must support realtime clocks.")

    def create_run_context(self, *, spec, adapter, host, model_warmup_plan):
        del spec, adapter
        return RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy
            ),
            model_warmup_plan=model_warmup_plan,
        )

    def create_session_edges(self, *, context, spec, scenario, provider, adapter):
        del spec, scenario, provider, adapter
        return SessionEdges(
            input_source=self.input,
            output_sink=self.output,
            cleanup_tasks=context.cleanup_tasks,
            error_policy=NativeWindowErrorPolicy(),
            transport=self.transport,
            clock=ResamplerRealtimeClock(resampler=self.input.resampler),
            activation=AlwaysActiveActivationPolicy(anchor_clock=True),
        )

    def select_driver(self) -> RealtimeSessionDriver:
        return RealtimeSessionDriver()


__all__ = [
    "NativeFrameQueue",
    "NativeWindowInputSource",
    "NativeWindowOutputSink",
    "NativeWindowRunMode",
]
