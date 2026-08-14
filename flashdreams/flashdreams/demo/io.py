# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport-neutral application output and factory contracts."""

from __future__ import annotations

import math
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Literal, Protocol, runtime_checkable

from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.canonical import (
    DeviceConverter,
    GamepadToDriverCommand,
    InputCanonicalizer,
    KeyboardToDriverCommand,
)
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.output import OutputArtifact

_EMPTY_CANONICAL_INPUT_SCHEMA = CanonicalInputSchema()
_EMPTY_USER_INPUT_SCHEMA = UserInputSchema()


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionInfo:
    """Sink-facing metadata known after application session initialization."""

    output_layout: str | None = None
    """Declared tensor layout for generated video results."""

    steady_output_frame_count: int | None = None
    """Expected frame count for steady-state output chunks."""

    frames_per_second: float | None = None
    """Presentation frame rate, when the application produces timed media."""

    video_width: int | None = None
    """Output video width in pixels, when known."""

    video_height: int | None = None
    """Output video height in pixels, when known."""

    metadata: Mapping[str, object] = field(default_factory=dict)
    """Additional immutable application metadata for sink setup."""

    def __post_init__(self) -> None:
        if self.output_layout is not None and not self.output_layout.strip():
            raise ValueError("SessionInfo.output_layout must be non-empty when set.")
        if (
            self.steady_output_frame_count is not None
            and self.steady_output_frame_count < 0
        ):
            raise ValueError(
                "SessionInfo.steady_output_frame_count must be >= 0 when set."
            )
        if self.frames_per_second is not None and (
            not math.isfinite(self.frames_per_second) or self.frames_per_second <= 0
        ):
            raise ValueError("SessionInfo.frames_per_second must be > 0 when set.")
        if self.video_width is not None and self.video_width <= 0:
            raise ValueError("SessionInfo.video_width must be > 0 when set.")
        if self.video_height is not None and self.video_height <= 0:
            raise ValueError("SessionInfo.video_height must be > 0 when set.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class OutputDecision:
    """Flow-control decision returned after one output write."""

    should_stop: bool = False
    """Whether the application should stop generating this session."""

    dropped: bool = False
    """Whether the sink dropped the submitted output chunk."""

    drop_policy: Literal["none", "drop_newest", "drop_oldest"] = "none"
    """Queue policy responsible for a dropped chunk."""

    backpressure_s: float = 0.0
    """Pacing delay for a realtime driver to account for before the next step."""

    metadata: Mapping[str, object] = field(default_factory=dict)
    """Sink-specific immutable delivery metadata."""

    def __post_init__(self) -> None:
        if self.drop_policy not in {"none", "drop_newest", "drop_oldest"}:
            raise ValueError(f"Unsupported drop_policy={self.drop_policy!r}.")
        if not math.isfinite(self.backpressure_s) or self.backpressure_s < 0:
            raise ValueError("OutputDecision.backpressure_s must be finite and >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class InputHandler:
    """Canonicalize transport-neutral user events for one application session."""

    def __init__(
        self,
        input_schema: CanonicalInputSchema = _EMPTY_CANONICAL_INPUT_SCHEMA,
        *,
        source_schema: UserInputSchema = _EMPTY_USER_INPUT_SCHEMA,
        converters: Iterable[DeviceConverter] | None = None,
        poll_events: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a handler for one canonical input schema.

        Args:
            input_schema: Canonical modalities requested by the application.
            source_schema: Raw event capabilities provided by the bound backend.
            converters: Device converters; ``None`` uses the standard registry.
            poll_events: Optional callback that asks the backend to emit events.
            clock: Monotonic clock used for session-relative event timestamps.

        Raises:
            ValueError: The source cannot provide a requested modality.
        """
        if converters is None:
            converters = (KeyboardToDriverCommand(), GamepadToDriverCommand())
        self._canonicalizer = InputCanonicalizer(converters)
        available_schema = self._canonicalizer.canonical_schema(source_schema)
        unsupported = [
            modality.name
            for modality in input_schema.modalities
            if not available_schema.supports(modality)
        ]
        if unsupported:
            raise ValueError(
                "Input source cannot provide canonical modalities: "
                f"{sorted(set(unsupported))}."
            )

        self._requested_names = frozenset(
            modality.name for modality in input_schema.modalities
        )
        relevant_converters = (
            converter
            for converter in self._canonicalizer.converters_for(source_schema)
            if converter.schema.produces.name in self._requested_names
        )
        self._accepted_event_types = frozenset(
            capability.event_type
            for converter in relevant_converters
            for capability in converter.schema.consumes
        )
        self._source_schema = source_schema
        self._poll_events = poll_events
        self._clock = clock
        self._events: list[UserInputEvent] = []
        self._lock = Lock()
        self._session_start_s = 0.0
        self._window_start_s = 0.0
        self._opened = False

    @property
    def accepts_events(self) -> bool:
        """Return whether the application consumes any backend event type."""
        return bool(self._accepted_event_types)

    def accepts_event_type(self, event_type: str) -> bool:
        """Return whether ``event_type`` can feed a requested modality."""
        return event_type in self._accepted_event_types

    def open(self, session_info: SessionInfo) -> None:
        """Open the handler and reset input state for one session."""
        del session_info
        self._canonicalizer.reset()
        with self._lock:
            self._events.clear()
            self._session_start_s = self._clock()
            self._window_start_s = 0.0
            self._opened = True

    def session_time_s(self) -> float:
        """Return the current session-relative event timestamp."""
        with self._lock:
            if not self._opened:
                return 0.0
            session_start_s = self._session_start_s
        return max(0.0, self._clock() - session_start_s)

    def submit_event(self, event: UserInputEvent) -> bool:
        """Queue one normalized event; return whether it was accepted.

        Events racing with a window drain are folded into the next window so a
        delayed backend callback cannot lose an input edge.
        """
        if event.event_type not in self._accepted_event_types:
            return False
        self._source_schema.validate_event(event)
        with self._lock:
            if not self._opened:
                return False
            if event.timestamp_s < self._window_start_s:
                event = replace(event, timestamp_s=self._window_start_s)
            self._events.append(event)
        return True

    def current_inputs(self) -> CanonicalInputWindow:
        """Poll events and return canonical inputs for the elapsed window."""
        with self._lock:
            if not self._opened:
                raise RuntimeError("Cannot fetch inputs from a closed input handler.")
        if self._poll_events is not None:
            self._poll_events()

        now_s = self.session_time_s()
        with self._lock:
            if not self._opened:
                raise RuntimeError("Cannot fetch inputs from a closed input handler.")
            now_s = max(now_s, self._window_start_s)
            events = tuple(sorted(self._events, key=lambda event: event.timestamp_s))
            self._events.clear()
            if events and events[-1].timestamp_s >= now_s:
                now_s = math.nextafter(events[-1].timestamp_s, math.inf)
            window = TimeWindow(start_s=self._window_start_s, end_s=now_s)
            self._window_start_s = now_s
        canonical = self._canonicalizer.canonicalize(
            UserInputs(events=events),
            window=window,
            source_schema=self._source_schema,
        )
        return CanonicalInputWindow(
            values={
                name: value
                for name, value in canonical.values.items()
                if name in self._requested_names
            },
            metadata=canonical.metadata,
            window=window,
        )

    def close(self) -> None:
        """Close the handler and discard queued events."""
        with self._lock:
            self._opened = False
            self._events.clear()


@runtime_checkable
class OutputSink(Protocol):
    """Consume canonical generated results for one application session."""

    produces_artifacts: bool
    """Whether closing the sink can produce persistent artifacts."""

    @abstractmethod
    def open(self, session_info: SessionInfo) -> None:
        """Prepare output resources for a session."""
        ...

    @abstractmethod
    def begin_generation(self, generation: int) -> None:
        """Start a generation and discard stale live output when required."""
        ...

    @abstractmethod
    def write(self, result: StepResult) -> OutputDecision:
        """Consume one result and return flow-control state."""
        ...

    @abstractmethod
    def close(self) -> Sequence[OutputArtifact]:
        """Finalize output resources and return persistent artifacts."""
        ...


@runtime_checkable
class IOFactory(Protocol):
    """Create isolated application input handling and output delivery."""

    @abstractmethod
    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create a handler for the application-declared canonical inputs."""
        ...

    @abstractmethod
    def create_output_sink(self) -> OutputSink:
        """Create the output sink for one application run."""
        ...


__all__ = [
    "IOFactory",
    "InputHandler",
    "OutputDecision",
    "OutputSink",
    "SessionInfo",
]
