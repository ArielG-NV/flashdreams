# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application I/O adapters for the shared asynchronous presenter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from flashdreams.demo.io import OutputDecision, OutputSink, SessionInfo
from flashdreams.runtime.inputs import UserInputEvent, UserInputSchema
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.presentation import (
    AsyncPresentationCoordinator,
    PresentationFrame,
    TorchAlphaCompositor,
)
from flashdreams.runtime.types import StepResult
from flashdreams.runtime.ui_input import IMGUI_RAW_INPUT_SCHEMA


@dataclass(slots=True)
class AsyncPresentationOutputSink(OutputSink):
    """Expose a frame-paced presentation coordinator as an output sink."""

    coordinator_factory: Callable[[SessionInfo], AsyncPresentationCoordinator]
    produces_artifacts: bool = True
    _coordinator: AsyncPresentationCoordinator | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def open(self, session_info: SessionInfo) -> None:
        coordinator = self.coordinator_factory(session_info)
        coordinator.open()
        self._coordinator = coordinator

    def begin_generation(self, generation: int) -> None:
        coordinator = self._require_coordinator()
        coordinator.begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        submission = self._require_coordinator().submit(result)
        return OutputDecision(
            dropped=submission.replaced_chunks > 0,
            drop_policy=("drop_oldest" if submission.replaced_chunks > 0 else "none"),
            backpressure_s=submission.queued_duration_s,
            metadata={
                "presentation_backend": "async-composited",
                "replaced_pending_chunks": submission.replaced_chunks,
            },
        )

    def close(self) -> Sequence[OutputArtifact]:
        coordinator = self._coordinator
        self._coordinator = None
        return () if coordinator is None else coordinator.close()

    def _require_coordinator(self) -> AsyncPresentationCoordinator:
        coordinator = self._coordinator
        if coordinator is None:
            raise RuntimeError("Cannot use a closed async presentation sink.")
        return coordinator


@dataclass(slots=True)
class ServerUIPresentationOutputSink(OutputSink):
    """Enable async server-side ImGui composition when a session declares UI."""

    sink: OutputSink
    bind_raw_input: Callable[[Callable[[UserInputEvent], None] | None], None] | None = (
        field(default=None, repr=False)
    )
    source_schema: UserInputSchema = field(
        default_factory=lambda: IMGUI_RAW_INPUT_SCHEMA,
        repr=False,
    )
    server_ui_opened: Callable[[object], None] | None = field(
        default=None,
        repr=False,
    )
    renderer_factory: Callable[..., Any] | None = field(default=None, repr=False)
    max_pending_chunks: int = 2
    produces_artifacts: bool = field(init=False)
    _coordinator: AsyncPresentationCoordinator | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _direct: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.produces_artifacts = self.sink.produces_artifacts

    def open(self, session_info: SessionInfo) -> None:
        """Open the original sink or its async ImGui presentation decorator."""
        server_ui = session_info.server_ui
        if server_ui is None:
            self.sink.open(session_info)
            self._direct = True
            return
        if (
            session_info.metadata.get("awaiting_ui_submission") is True
            and self.bind_raw_input is None
        ):
            raise ValueError(
                "This output cannot collect the initial UI prompt; provide --prompt."
            )
        if self.server_ui_opened is not None:
            self.server_ui_opened(server_ui)
        if (
            session_info.frames_per_second is None
            or session_info.video_width is None
            or session_info.video_height is None
        ):
            raise ValueError(
                "Server UI requires session fps, video_width, and video_height."
            )

        if self.renderer_factory is None:
            from flashdreams.runtime.imgui import create_slangpy_imgui_renderer

            renderer_factory = create_slangpy_imgui_renderer
        else:
            renderer_factory = self.renderer_factory
        renderer = renderer_factory(
            width=session_info.video_width,
            height=session_info.video_height,
            source_schema=self.source_schema,
            build_ui=server_ui.build_ui,
            controls=server_ui.controls,
        )
        idle_frame = None
        if self.bind_raw_input is not None:
            import numpy as np

            idle_frame = np.zeros(
                (session_info.video_height, session_info.video_width, 3),
                dtype=np.uint8,
            )
        backend = OutputSinkPresentationBackend(
            sink=self.sink,
            session_info=session_info,
            frame_to_result=_presentation_frame_to_result,
        )
        coordinator = AsyncPresentationCoordinator(
            fps=session_info.frames_per_second,
            ui_renderer=renderer,
            compositor=TorchAlphaCompositor(),
            backends=(backend,),
            max_pending_chunks=self.max_pending_chunks,
            idle_frame=idle_frame,
        )
        if self.bind_raw_input is not None:
            self.bind_raw_input(renderer.publish_raw_input)
        try:
            coordinator.open()
        except BaseException:
            if self.bind_raw_input is not None:
                self.bind_raw_input(None)
            raise
        self._coordinator = coordinator

    def begin_generation(self, generation: int) -> None:
        """Begin a generation on the active presentation path."""
        if self._direct:
            self.sink.begin_generation(generation)
            return
        self._require_coordinator().begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        """Queue a UI-composited chunk without blocking model generation."""
        if self._direct:
            return self.sink.write(result)
        submission = self._require_coordinator().submit(result)
        return OutputDecision(
            dropped=submission.replaced_chunks > 0,
            drop_policy=("drop_oldest" if submission.replaced_chunks > 0 else "none"),
            backpressure_s=submission.queued_duration_s,
            metadata={
                "presentation_backend": "async-imgui",
                "replaced_pending_chunks": submission.replaced_chunks,
            },
        )

    def close(self) -> Sequence[OutputArtifact]:
        """Close the active path after draining its presentation queue."""
        if self._direct:
            self._direct = False
            return self.sink.close()
        coordinator = self._coordinator
        self._coordinator = None
        if coordinator is None:
            return ()
        try:
            return coordinator.close()
        finally:
            if self.bind_raw_input is not None:
                self.bind_raw_input(None)

    def _require_coordinator(self) -> AsyncPresentationCoordinator:
        coordinator = self._coordinator
        if coordinator is None:
            raise RuntimeError("Cannot use a closed server UI presentation sink.")
        return coordinator


def _presentation_frame_to_result(frame: PresentationFrame) -> StepResult:
    """Wrap one composited HWC frame as a single-frame TCHW result."""
    import numpy as np
    import torch

    value = frame.frame
    if not torch.is_tensor(value):
        value = torch.as_tensor(np.ascontiguousarray(value))
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Composited presentation frames must use [H, W, RGB] layout.")
    chunk = value.permute(2, 0, 1).unsqueeze(0).contiguous()
    return StepResult.from_video_chunk(
        step_index=frame.step_index,
        video_chunk=chunk,
        layout="tchw",
        metadata=frame.metadata,
    )


@dataclass(slots=True)
class LocalWindowPresentationBackend:
    """Present final composited pixels directly on the presentation thread."""

    presenter_factory: Callable[..., Any]
    presenter_kwargs: Mapping[str, object]
    presenter_opened: Callable[[Any], None] | None = None
    _presenter: Any | None = field(default=None, init=False, repr=False)

    def open(self) -> None:
        presenter = self.presenter_factory(**dict(self.presenter_kwargs))
        self._presenter = presenter
        if self.presenter_opened is not None:
            self.presenter_opened(presenter)

    def present(self, frame: PresentationFrame) -> None:
        presenter = self._presenter
        if presenter is None:
            raise RuntimeError("Local presentation backend is closed.")
        process_events = getattr(presenter, "process_events", None)
        if callable(process_events):
            process_events()
        if not presenter.present(frame.frame):
            raise RuntimeError("Local presentation window closed.")

    def close(self) -> Sequence[OutputArtifact]:
        presenter = self._presenter
        self._presenter = None
        if presenter is not None:
            presenter.close()
        return ()


@dataclass(slots=True)
class OutputSinkPresentationBackend:
    """Adapt WebRTC, MP4, or another result sink to composited frames.

    ``frame_to_result`` converts the final composited pixel object to the
    layout expected by the wrapped sink. This keeps encoding and persistence
    details in the existing target implementations while all targets receive
    the same :class:`PresentationFrame` first.
    """

    sink: OutputSink
    session_info: SessionInfo
    frame_to_result: Callable[[PresentationFrame], StepResult]
    _generation: int | None = field(default=None, init=False, repr=False)

    def open(self) -> None:
        self.sink.open(self.session_info)

    def present(self, frame: PresentationFrame) -> None:
        if frame.generation != self._generation:
            self.sink.begin_generation(frame.generation)
            self._generation = frame.generation
        decision = self.sink.write(self.frame_to_result(frame))
        if decision.should_stop:
            raise RuntimeError("Wrapped presentation output requested stop.")

    def close(self) -> Sequence[OutputArtifact]:
        return self.sink.close()


__all__ = [
    "AsyncPresentationOutputSink",
    "LocalWindowPresentationBackend",
    "ServerUIPresentationOutputSink",
    "OutputSinkPresentationBackend",
]
