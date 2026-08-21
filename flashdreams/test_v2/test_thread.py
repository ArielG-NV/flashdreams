# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for v2 session workers and event fan-out."""

import threading
from typing import Any

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2 import imgui_thread
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _Thread(IThread[None]):
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        raise NotImplementedError

    def reset(self) -> None:
        return


@pytest.mark.parametrize("frequency", [-1, 1.5, True])
def test_thread_rejects_non_uint_frequencies(frequency: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="frequency"):
        _Thread(state=None, frequency=frequency)


def test_zero_frequency_is_accepted_as_unbounded() -> None:
    assert _Thread(state=None, frequency=0).frequency == 0


def test_frequency_sets_a_minimum_interval_between_step_starts() -> None:
    thread = _Thread(state=None, frequency=20)
    stop = threading.Event()
    first_start = thread._pace(None, stop)
    second_start = thread._pace(first_start, stop)

    assert second_start - first_start >= 0.04


def test_event_buffer_fans_out_and_collects_shared_prefix() -> None:
    buffer = EventBuffer()
    buffer.read(0)
    buffer.read(1)
    event = UserInputEvent(
        timestamp=uint64(1),
        event_data=KeyboardUserInputEventData(key="x", pressed=True),
    )
    buffer.append(UserInputEvents([event]))

    events_zero, _ = buffer.read(0)
    assert events_zero.get_events() == [event]
    assert buffer.collect_garbage() == 0
    events_one, _ = buffer.read(1)
    assert events_one.get_events() == [event]
    assert buffer.collect_garbage() == 1
    assert buffer.retained_count() == 0


def test_event_buffer_clear_removes_events_and_readers() -> None:
    buffer = EventBuffer()
    buffer.read(0)
    event = UserInputEvent(
        timestamp=uint64(1),
        event_data=KeyboardUserInputEventData(key="x", pressed=True),
    )
    buffer.append(UserInputEvents([event]))

    buffer.clear()

    assert buffer.retained_count() == 0
    assert buffer.collect_garbage() == 0
    events, _ = buffer.read(0)
    assert events.get_events() == []


def test_event_buffer_unregister_stops_retaining_for_finished_reader() -> None:
    buffer = EventBuffer()
    buffer.read(0)
    buffer.read(1)
    event = UserInputEvent(
        timestamp=uint64(1),
        event_data=KeyboardUserInputEventData(key="x", pressed=True),
    )
    buffer.append(UserInputEvents([event]))
    buffer.read(0)

    buffer.unregister(1)

    assert buffer.collect_garbage() == 1


class _Renderer:
    def __init__(self, *, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.waited = False
        self.was_reset = False

    def render(self, events: UserInputEvents, draw_ui: Any) -> torch.Tensor:
        del events
        draw_ui(object())
        return torch.zeros((2, 2, 4))

    def wait_for_cuda_frame(self, frame: torch.Tensor) -> torch.Tensor:
        self.waited = True
        return frame.permute(2, 0, 1).unsqueeze(0) + 1

    def reset(self) -> None:
        self.was_reset = True

    def close(self) -> None:
        return


class _UIThread(ImGUIThread[None]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.draw_calls: list[tuple[Any, int, UserInputEvents]] = []

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> None:
        self.draw_calls.append((imgui, step_index, events))


def test_imgui_thread_wraps_one_rendered_frame_in_a_step_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(imgui_thread, "SlangPyImGUIRenderer", _Renderer)
    thread = _UIThread(
        state=None,
        frequency=60,
        output_layout=VideoTensorLayout.tchw,
        width=2,
        height=2,
    )

    events = UserInputEvents([])
    result = thread.step(4, events)

    assert thread._renderer.waited
    assert thread._renderer.width == 2
    assert thread._renderer.height == 2
    assert len(thread.draw_calls) == 1
    _, draw_step_index, draw_events = thread.draw_calls[0]
    assert draw_step_index == 4
    assert draw_events is events
    assert result.step_index == 4
    assert result.frame_count == 1
    assert result.output_layout is VideoTensorLayout.tchw
    assert torch.all(result.output == 1)

    thread.reset()
    assert thread._renderer.was_reset
