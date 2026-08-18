# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for frame-paced UI composition and output fanout."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import cast

import pytest
import torch

from flashdreams.demo import (
    NullOutputSink,
    OutputDecision,
    ServerUIPresentationOutputSink,
    SessionInfo,
)
from flashdreams.runtime import (
    SERVER_UI_CLOSE_CONTROL_ID,
    AsyncPresentationCoordinator,
    OutputArtifact,
    PresentationFrame,
    PresentationStopRequested,
    ServerUI,
    StepResult,
    UIControlMailbox,
)

pytestmark = pytest.mark.ci_cpu


class _UIRenderer:
    def __init__(self, controls: UIControlMailbox) -> None:
        self.controls = controls
        self.threads: list[int] = []
        self.closed = False

    def render_ui(
        self,
        *,
        presentation_index: int,
        presentation_time_s: float,
    ) -> object:
        del presentation_index, presentation_time_s
        self.threads.append(threading.get_ident())
        return self.controls.snapshot(consume_events=False).values["version"]

    def close(self) -> None:
        self.closed = True


class _Compositor:
    def composite(self, video_frame: object, ui_frame: object) -> object:
        return video_frame, ui_frame


class _Backend:
    def __init__(self, controls: UIControlMailbox, expected_frames: int) -> None:
        self.controls = controls
        self.expected_frames = expected_frames
        self.frames: list[PresentationFrame] = []
        self.done = threading.Event()
        self.open_thread: int | None = None
        self.close_thread: int | None = None

    def open(self) -> None:
        self.open_thread = threading.get_ident()

    def present(self, frame: PresentationFrame) -> None:
        self.frames.append(frame)
        if frame.presentation_index == 0:
            self.controls.set_value("version", 2)
        elif frame.presentation_index == 2:
            self.controls.set_value("version", 3)
        if len(self.frames) == self.expected_frames:
            self.done.set()

    def close(self) -> Sequence[OutputArtifact]:
        self.close_thread = threading.get_ident()
        return ()


def test_ui_is_rendered_per_frame_while_model_thread_is_free() -> None:
    controls = UIControlMailbox()
    controls.set_value("version", 1)
    ui = _UIRenderer(controls)
    first = _Backend(controls, expected_frames=5)
    second = _Backend(controls, expected_frames=5)
    coordinator = AsyncPresentationCoordinator(
        fps=240.0,
        ui_renderer=ui,
        compositor=_Compositor(),
        backends=(first, second),
    )
    coordinator.open()
    model_thread = threading.get_ident()

    submission = coordinator.submit(
        StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((5, 3, 2, 2)),
            layout="tchw",
        )
    )

    assert submission.replaced_chunks == 0
    coordinator.close()
    assert first.done.is_set()

    assert [cast(tuple[object, int], frame.frame)[1] for frame in first.frames] == [
        1,
        2,
        2,
        3,
        3,
    ]
    assert [frame.frame for frame in second.frames] == [
        frame.frame for frame in first.frames
    ]
    assert all(thread_id != model_thread for thread_id in ui.threads)
    assert first.open_thread == first.close_thread == ui.threads[0]
    assert ui.closed


def test_idle_frame_keeps_ui_rendering_without_model_output() -> None:
    controls = UIControlMailbox()
    controls.set_value("version", 1)
    ui = _UIRenderer(controls)
    backend = _Backend(controls, expected_frames=2)
    coordinator = AsyncPresentationCoordinator(
        fps=240.0,
        ui_renderer=ui,
        compositor=_Compositor(),
        backends=(backend,),
        idle_frame="waiting-for-model",
    )

    coordinator.open()
    assert backend.done.wait(timeout=1.0)
    coordinator.close()

    first, second = backend.frames[:2]
    assert first.frame == ("waiting-for-model", 1)
    assert second.frame == ("waiting-for-model", 2)
    assert first.metadata["presentation_idle"] is True
    assert all(thread_id == backend.open_thread for thread_id in ui.threads)


def test_presentation_stop_wakes_owner_and_rejects_later_submissions() -> None:
    controls = UIControlMailbox()
    controls.set_value("version", 1)
    stopped = threading.Event()

    class _ClosedBackend(_Backend):
        def present(self, frame: PresentationFrame) -> None:
            del frame
            raise PresentationStopRequested("window closed")

    coordinator = AsyncPresentationCoordinator(
        fps=240.0,
        ui_renderer=_UIRenderer(controls),
        compositor=_Compositor(),
        backends=(_ClosedBackend(controls, expected_frames=0),),
        idle_frame="waiting-for-model",
        on_stop_requested=stopped.set,
    )

    coordinator.open()
    assert stopped.wait(timeout=1.0)
    submission = coordinator.submit(
        StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((1, 3, 2, 2)),
            layout="tchw",
        )
    )
    coordinator.close()

    assert submission.should_stop


def test_ui_control_mailbox_wait_consumes_only_matching_events() -> None:
    controls = UIControlMailbox(clock=lambda: 10.0)
    unrelated = controls.emit("cancel")
    expected = controls.emit("generate", "a lighthouse")

    actual = controls.wait_for_event("generate")
    remaining = controls.snapshot()

    assert actual == expected
    assert remaining.events == (unrelated,)


def test_ui_control_mailbox_waits_for_any_requested_event() -> None:
    controls = UIControlMailbox(clock=lambda: 10.0)
    controls.emit("unrelated")
    expected = controls.emit("close")

    actual = controls.wait_for_any(("generate", "close"))
    remaining = controls.snapshot()

    assert actual == expected
    assert [event.control_id for event in remaining.events] == ["unrelated"]


def test_ui_control_mailbox_gives_step_atomic_state_and_clicks() -> None:
    controls = UIControlMailbox(clock=lambda: 10.0)
    controls.set_value("prompt", "a lighthouse")
    click = controls.emit("generate")

    first_step = controls.snapshot()
    second_step = controls.snapshot()

    assert first_step.values == {"prompt": "a lighthouse"}
    assert first_step.events == (click,)
    assert second_step.values == first_step.values
    assert second_step.events == ()


def test_noninteractive_sink_rejects_a_session_waiting_for_ui_input() -> None:
    wrapped = ServerUIPresentationOutputSink(sink=NullOutputSink())

    with pytest.raises(ValueError, match="provide --prompt"):
        wrapped.open(
            SessionInfo(
                frames_per_second=24.0,
                video_width=2,
                video_height=2,
                metadata={"awaiting_ui_submission": True},
                server_ui=ServerUI(build_ui=lambda _imgui, _controls: None),
            )
        )


class _AlphaRenderer:
    def __init__(self) -> None:
        self.raw_inputs: list[object] = []
        self.closed = False

    def publish_raw_input(self, event: object) -> None:
        self.raw_inputs.append(event)

    def render_ui(
        self,
        *,
        presentation_index: int,
        presentation_time_s: float,
    ) -> object:
        del presentation_index, presentation_time_s
        layer = torch.zeros((2, 2, 4), dtype=torch.uint8)
        layer[..., 0] = 255
        layer[..., 3] = 128
        return layer

    def close(self) -> None:
        self.closed = True


def test_server_ui_sink_composites_and_forwards_final_frames() -> None:
    renderer = _AlphaRenderer()
    bindings: list[object] = []
    sink = NullOutputSink(store_outputs=True)
    server_ui = ServerUI(build_ui=lambda _imgui, _controls: None)
    wrapped = ServerUIPresentationOutputSink(
        sink=sink,
        bind_raw_input=bindings.append,
        renderer_factory=lambda **_kwargs: renderer,
    )
    wrapped.open(
        SessionInfo(
            output_layout="tchw",
            frames_per_second=240.0,
            video_width=2,
            video_height=2,
            server_ui=server_ui,
        )
    )
    wrapped.begin_generation(0)
    decision = wrapped.write(
        StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
            layout="tchw",
        )
    )
    wrapped.close()

    assert decision.metadata["presentation_backend"] == "async-imgui"
    assert renderer.closed
    assert callable(bindings[0])
    assert bindings[-1] is None
    assert len(sink.outputs) == 1
    output = sink.outputs[0]
    assert tuple(output.shape) == (1, 3, 2, 2)
    assert torch.equal(output[0, :, 0, 0], torch.tensor([128, 0, 0]))


def test_server_ui_sink_publishes_close_when_output_requests_stop() -> None:
    class _StopSink(NullOutputSink):
        def write(self, result: StepResult) -> OutputDecision:
            super().write(result)
            return OutputDecision(should_stop=True)

    server_ui = ServerUI(build_ui=lambda _imgui, _controls: None)
    close_events: list[object] = []
    waiter = threading.Thread(
        target=lambda: close_events.append(
            server_ui.controls.wait_for_event(SERVER_UI_CLOSE_CONTROL_ID)
        ),
        daemon=True,
    )
    wrapped = ServerUIPresentationOutputSink(
        sink=_StopSink(),
        bind_raw_input=lambda _target: None,
        renderer_factory=lambda **_kwargs: _AlphaRenderer(),
    )

    waiter.start()
    wrapped.open(
        SessionInfo(
            output_layout="tchw",
            frames_per_second=240.0,
            video_width=2,
            video_height=2,
            server_ui=server_ui,
        )
    )
    waiter.join(timeout=1.0)
    wrapped.close()

    assert not waiter.is_alive()
    assert len(close_events) == 1
