# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-paced UI composition independent of model execution."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepResult

SERVER_UI_GENERATE_CONTROL_ID = "flashdreams.server_ui.generate"
"""Semantic UI event requesting a fresh model generation."""

SERVER_UI_CLOSE_CONTROL_ID = "flashdreams.server_ui.close"
"""Semantic UI event requesting graceful application shutdown."""


@dataclass(frozen=True, kw_only=True, slots=True)
class UIControlEvent:
    """One semantic UI action consumed by a later model step."""

    sequence: int
    timestamp_s: float
    control_id: str
    value: object = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0.")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and >= 0.")
        if not self.control_id.strip():
            raise ValueError("control_id must be non-empty.")


@dataclass(frozen=True, kw_only=True, slots=True)
class UIControlSnapshot:
    """Immutable control state and edge events visible to one model step."""

    revision: int
    values: Mapping[str, object] = field(default_factory=dict)
    events: Sequence[UIControlEvent] = ()

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("revision must be >= 0.")
        object.__setattr__(self, "values", freeze_mapping(self.values))
        object.__setattr__(self, "events", tuple(self.events))


class UIControlMailbox:
    """Share UI state with ``step`` without sharing mutable ImGui state."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._event_available = threading.Condition(self._lock)
        self._values: dict[str, object] = {}
        self._events: list[UIControlEvent] = []
        self._revision = 0
        self._next_sequence = 0
        self._start_s = clock()

    def set_value(self, control_id: str, value: object) -> None:
        """Publish the latest value of a level-triggered UI control."""
        if not control_id.strip():
            raise ValueError("control_id must be non-empty.")
        with self._lock:
            if self._values.get(control_id, object()) == value:
                return
            self._values[control_id] = value
            self._revision += 1

    def emit(self, control_id: str, value: object = None) -> UIControlEvent:
        """Publish one edge-triggered action such as a button click."""
        if not control_id.strip():
            raise ValueError("control_id must be non-empty.")
        with self._lock:
            event = UIControlEvent(
                sequence=self._next_sequence,
                timestamp_s=max(0.0, self._clock() - self._start_s),
                control_id=control_id,
                value=value,
            )
            self._next_sequence += 1
            self._events.append(event)
            self._revision += 1
            self._event_available.notify_all()
            return event

    def wait_for_event(self, control_id: str) -> UIControlEvent:
        """Wait for and consume the newest event matching ``control_id``."""
        return self.wait_for_any((control_id,))

    def wait_for_any(self, control_ids: Sequence[str]) -> UIControlEvent:
        """Wait for and consume the newest event matching any requested control."""
        requested = frozenset(control_ids)
        if not requested or any(not control_id.strip() for control_id in requested):
            raise ValueError("control_ids must contain non-empty identifiers.")
        with self._event_available:
            while not any(event.control_id in requested for event in self._events):
                self._event_available.wait()
            matching = [
                event for event in self._events if event.control_id in requested
            ]
            self._events[:] = [
                event for event in self._events if event.control_id not in requested
            ]
            return matching[-1]

    def snapshot(self, *, consume_events: bool = True) -> UIControlSnapshot:
        """Return an atomic snapshot for ``step`` and optionally drain edges."""
        with self._lock:
            snapshot = UIControlSnapshot(
                revision=self._revision,
                values=dict(self._values),
                events=tuple(self._events),
            )
            if consume_events:
                self._events.clear()
            return snapshot


@dataclass(kw_only=True, slots=True)
class ServerUI:
    """Per-session ImGui callback and its thread-safe model mailbox."""

    build_ui: Callable[[Any, UIControlMailbox], None]
    """Build one immediate-mode UI frame on the presentation thread."""

    controls: UIControlMailbox = field(default_factory=UIControlMailbox)
    """Semantic values and edge events consumed by ``step``."""


@dataclass(frozen=True, kw_only=True, slots=True)
class PresentationFrame:
    """One fully composited frame shared by every output backend."""

    frame: object
    generation: int
    step_index: int
    frame_index: int
    presentation_index: int
    presentation_time_s: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("generation", self.generation),
            ("step_index", self.step_index),
            ("frame_index", self.frame_index),
            ("presentation_index", self.presentation_index),
        ):
            if value < 0:
                raise ValueError(f"{label} must be >= 0.")
        if not math.isfinite(self.presentation_time_s) or self.presentation_time_s < 0:
            raise ValueError("presentation_time_s must be finite and >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class UIFrameRenderer(Protocol):
    """Render a fresh server-side UI layer on the presentation thread."""

    def render_ui(
        self,
        *,
        presentation_index: int,
        presentation_time_s: float,
    ) -> object: ...


@runtime_checkable
class FrameCompositor(Protocol):
    """Composite a generated video frame and the newest rendered UI layer."""

    def composite(self, video_frame: object, ui_frame: object) -> object: ...


class TorchAlphaCompositor:
    """Alpha-blend an RGBA UI layer onto an RGB frame on the frame's device."""

    def composite(self, video_frame: object, ui_frame: object) -> object:
        """Return one uint8 HWC RGB tensor."""
        import numpy as np
        import torch

        to_cuda_tensor = getattr(video_frame, "to_cuda_tensor", None)
        if callable(to_cuda_tensor):
            video = to_cuda_tensor()
        else:
            to_numpy = getattr(video_frame, "to_numpy", None)
            video_value = to_numpy() if callable(to_numpy) else video_frame
            video = torch.as_tensor(np.ascontiguousarray(video_value))

        ui_to_numpy = getattr(ui_frame, "to_numpy", None)
        ui_value = ui_to_numpy() if callable(ui_to_numpy) else ui_frame
        ui = torch.as_tensor(
            np.ascontiguousarray(ui_value),
            device=video.device,
        )
        if video.ndim != 3 or video.shape[-1] != 3:
            raise ValueError(
                "Video frames must use uint8 [H, W, RGB] layout, "
                f"got {tuple(video.shape)}."
            )
        if ui.ndim != 3 or ui.shape[-1] != 4:
            raise ValueError(
                f"UI frames must use uint8 [H, W, RGBA] layout, got {tuple(ui.shape)}."
            )
        if tuple(ui.shape[:2]) != tuple(video.shape[:2]):
            raise ValueError(
                "UI and video frame dimensions must match: "
                f"{tuple(ui.shape[:2])} != {tuple(video.shape[:2])}."
            )

        video_i32 = video.to(dtype=torch.int32)
        ui_i32 = ui.to(dtype=torch.int32)
        alpha = ui_i32[..., 3:4]
        return ((video_i32 * (255 - alpha) + ui_i32[..., :3] * alpha + 127) // 255).to(
            dtype=torch.uint8
        )


@runtime_checkable
class PresentationBackend(Protocol):
    """Consume the same final composited frame for a concrete destination."""

    def open(self) -> None: ...

    def present(self, frame: PresentationFrame) -> None: ...

    def close(self) -> Sequence[OutputArtifact]: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class PresentationSubmission:
    """Queue result returned immediately to the model thread."""

    replaced_chunks: int = 0
    queued_duration_s: float = 0.0
    should_stop: bool = False


class PresentationStopRequested(RuntimeError):
    """Signal that a presentation backend was closed by its consumer."""


@dataclass(frozen=True, slots=True)
class _QueuedChunk:
    generation: int
    step_index: int
    frames: Sequence[object]
    metadata: Mapping[str, object]


class AsyncPresentationCoordinator:
    """Render UI, composite, and fan out frames independently of ``step``.

    ``submit`` only queues lazy frame handles. One presentation-owned thread
    then renders a new UI layer for every video frame, composites it once, and
    sends the identical final frame to local-window, WebRTC, MP4, or future
    :class:`PresentationBackend` implementations.
    """

    def __init__(
        self,
        *,
        fps: float,
        ui_renderer: UIFrameRenderer,
        source_fps: float | None = None,
        compositor: FrameCompositor,
        backends: Sequence[PresentationBackend],
        max_pending_chunks: int = 2,
        idle_frame: object | None = None,
        on_stop_requested: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and > 0.")
        if source_fps is None:
            source_fps = fps
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("source_fps must be finite and > 0.")
        if max_pending_chunks <= 0:
            raise ValueError("max_pending_chunks must be > 0.")
        self._frame_interval_s = 1.0 / fps
        self._fps = fps
        self._source_fps = source_fps
        self._presentations_per_source_frame = fps / source_fps
        self._ui_renderer = ui_renderer
        self._compositor = compositor
        self._backends = tuple(backends)
        self._clock = clock
        self._pending: queue.Queue[_QueuedChunk] = queue.Queue(max_pending_chunks)
        self._initial_idle_frame = idle_frame
        self._on_stop_requested = on_stop_requested
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._state_lock = threading.Lock()
        self._generation = 0
        self._error: BaseException | None = None
        self._artifacts: tuple[OutputArtifact, ...] = ()
        self._thread = threading.Thread(
            target=self._run,
            name="flashdreams-presentation",
            daemon=True,
        )

    def open(self) -> None:
        """Start every backend and the presentation-owned thread."""
        self._thread.start()
        self._ready.wait()
        self.raise_if_failed()

    def begin_generation(self, generation: int) -> None:
        """Discard queued chunks from older model generations."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        with self._state_lock:
            self._generation = generation
        self._discard_pending()

    def submit(self, result: StepResult) -> PresentationSubmission:
        """Queue one generated chunk and return without presenting it."""
        self.raise_if_failed()
        if self._stop.is_set():
            return PresentationSubmission(should_stop=True)
        if result.layout is None:
            raise TypeError("Async presentation requires a video StepResult.")
        with self._state_lock:
            generation = self._generation
        item = _QueuedChunk(
            generation=generation,
            step_index=result.step_index,
            frames=tuple(result.lazy_rgb_frames()),
            metadata=result.metadata,
        )
        replaced = 0
        self._idle.clear()
        while True:
            try:
                self._pending.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._pending.get_nowait()
                except queue.Empty:
                    continue
                replaced += 1
        queued_frames = self._pending.qsize() * len(item.frames)
        return PresentationSubmission(
            replaced_chunks=replaced,
            queued_duration_s=queued_frames / self._source_fps,
            should_stop=self._stop.is_set(),
        )

    def close(self) -> Sequence[OutputArtifact]:
        """Stop presentation and return artifacts from every backend."""
        while not self._idle.is_set() or not self._pending.empty():
            self._idle.wait(timeout=0.01)
            self.raise_if_failed()
        self._stop.set()
        self._thread.join()
        self.raise_if_failed()
        return self._artifacts

    def raise_if_failed(self) -> None:
        """Propagate a presentation-thread failure to the caller."""
        with self._state_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Asynchronous presentation failed.") from error

    def _run(self) -> None:
        opened: list[PresentationBackend] = []
        artifacts: list[OutputArtifact] = []
        try:
            for backend in self._backends:
                backend.open()
                opened.append(backend)
            self._ready.set()
            presentation_index = 0
            source_frame_index = 0
            active_generation = self._generation
            session_start_s = self._clock()
            held_video_frame = self._initial_idle_frame
            held_step_index = 0
            held_frame_index = 0
            held_metadata: Mapping[str, object] = {"presentation_idle": True}
            while not self._stop.is_set():
                is_idle_repeat = False
                try:
                    chunk = self._pending.get(timeout=0.001)
                except queue.Empty:
                    if held_video_frame is None:
                        continue
                    with self._state_lock:
                        generation = self._generation
                    chunk = _QueuedChunk(
                        generation=generation,
                        step_index=held_step_index,
                        frames=(held_video_frame,),
                        metadata=held_metadata,
                    )
                    is_idle_repeat = True
                if self._is_stale(chunk.generation):
                    if self._pending.empty():
                        self._idle.set()
                    continue
                if chunk.generation != active_generation:
                    active_generation = chunk.generation
                    source_frame_index = 0
                for queued_frame_index, video_frame in enumerate(chunk.frames):
                    if self._is_stale(chunk.generation):
                        break
                    frame_index = (
                        held_frame_index if is_idle_repeat else queued_frame_index
                    )
                    if not is_idle_repeat:
                        held_video_frame = video_frame
                        held_step_index = chunk.step_index
                        held_frame_index = frame_index
                        held_metadata = dict(chunk.metadata)
                        held_metadata["presentation_idle"] = True
                    repeat_count = (
                        1
                        if is_idle_repeat
                        else self._presentation_count(source_frame_index)
                    )
                    if not is_idle_repeat:
                        source_frame_index += 1
                    for repeat_index in range(repeat_count):
                        presentation_time_s = max(0.0, self._clock() - session_start_s)
                        ui_frame = self._ui_renderer.render_ui(
                            presentation_index=presentation_index,
                            presentation_time_s=presentation_time_s,
                        )
                        composited = self._compositor.composite(video_frame, ui_frame)
                        metadata = dict(chunk.metadata)
                        metadata["presentation_repeated"] = repeat_index > 0
                        presented = PresentationFrame(
                            frame=composited,
                            generation=chunk.generation,
                            step_index=chunk.step_index,
                            frame_index=frame_index,
                            presentation_index=presentation_index,
                            presentation_time_s=presentation_time_s,
                            metadata=metadata,
                        )
                        for backend in opened:
                            backend.present(presented)
                        presentation_index += 1
                        next_deadline_s = (
                            session_start_s
                            + presentation_index * self._frame_interval_s
                        )
                        self._stop.wait(max(0.0, next_deadline_s - self._clock()))
                if self._pending.empty():
                    self._idle.set()
        except PresentationStopRequested:
            self._stop.set()
            self._discard_pending()
            if self._on_stop_requested is not None:
                try:
                    self._on_stop_requested()
                except BaseException as exc:
                    with self._state_lock:
                        self._error = exc
        except BaseException as exc:
            with self._state_lock:
                self._error = exc
        finally:
            self._ready.set()
            self._idle.set()
            close_ui = getattr(self._ui_renderer, "close", None)
            if callable(close_ui):
                try:
                    close_ui()
                except BaseException as exc:
                    with self._state_lock:
                        if self._error is None:
                            self._error = exc
            for backend in reversed(opened):
                try:
                    artifacts.extend(backend.close())
                except BaseException as exc:
                    with self._state_lock:
                        if self._error is None:
                            self._error = exc
            self._artifacts = tuple(artifacts)

    def _is_stale(self, generation: int) -> bool:
        if self._stop.is_set():
            return True
        with self._state_lock:
            return generation != self._generation

    def _presentation_count(self, source_frame_index: int) -> int:
        """Return presentation ticks assigned to one source-frame interval."""
        ratio = self._presentations_per_source_frame
        start = math.ceil(source_frame_index * ratio)
        end = math.ceil((source_frame_index + 1) * ratio)
        return max(0, end - start)

    def _discard_pending(self) -> None:
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                self._idle.set()
                return


__all__ = [
    "AsyncPresentationCoordinator",
    "FrameCompositor",
    "TorchAlphaCompositor",
    "PresentationBackend",
    "PresentationFrame",
    "PresentationStopRequested",
    "PresentationSubmission",
    "SERVER_UI_CLOSE_CONTROL_ID",
    "ServerUI",
    "SERVER_UI_GENERATE_CONTROL_ID",
    "UIControlEvent",
    "UIControlMailbox",
    "UIControlSnapshot",
    "UIFrameRenderer",
]
