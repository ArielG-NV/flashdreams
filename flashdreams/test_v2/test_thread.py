# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the paced user-visible-thread contract."""

import queue
import threading
import time
from typing import cast

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.thread_manager import _ThreadManager
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _result(step_index: int) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32),
        frame_count=1,
        output_layout=VideoTensorLayout.bcthw,
    )


class RecordingThread(IThread[list[str]]):
    """Record reset, step, and message calls in typed list state."""

    def __init__(
        self,
        frequency: int,
    ) -> None:
        self.step_times: list[float] = []
        super().__init__(state=[], frequency=frequency)

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self.state.append(f"step:{step_index}:{len(events.get_events())}")
        self.step_times.append(time.monotonic())
        return _result(step_index)

    def reset(self) -> None:
        self.state.append("reset")


def _event(
    event_data: ResetUserInputEventData | CloseUserInputEventData,
) -> UserInputEvent:
    return UserInputEvent(timestamp=uint64(0), event_data=event_data)


def _manager_for(thread: RecordingThread) -> _ThreadManager:
    manager = _ThreadManager()
    manager._register_model_generation_thread(thread)
    return manager


def _run(
    manager: _ThreadManager,
    *,
    event_buffer: EventBuffer | None = None,
    max_steps: int | None = 1,
) -> queue.Queue[BaseException]:
    finished = threading.Event()
    failures: queue.Queue[BaseException] = queue.Queue()
    manager._start(
        event_buffer=event_buffer or EventBuffer(),
        stop=threading.Event(),
        failure=failures,
        finished=finished,
        max_steps=max_steps,
    )
    assert finished.wait(timeout=2)
    manager._stop(timeout_seconds=2)
    return failures


def test_invoke_async_runs_a_message_before_the_next_step() -> None:
    thread = RecordingThread(0)
    manager = _manager_for(thread)

    assert (
        thread.invoke_async(0, lambda thread_state: thread_state.append("message"))
        is None
    )
    failures = _run(manager)

    assert failures.empty()
    assert thread.state == ["message", "step:0:0"]
    assert thread.latest_step is not None
    assert thread.latest_step.step_index == 0


def test_message_cannot_return_a_value() -> None:
    thread = RecordingThread(0)
    manager = _manager_for(thread)

    def invalid_message(_thread_state: list[str]) -> None:
        return cast(None, "not allowed")

    thread.invoke_async(0, invalid_message)
    failures = _run(manager)

    error = failures.get_nowait()
    assert isinstance(error, TypeError)
    assert "must return None" in str(error)
    assert thread.state == []


def test_reset_event_resets_before_step_zero_with_the_whole_batch() -> None:
    event_buffer = EventBuffer()
    thread = RecordingThread(0)
    manager = _manager_for(thread)
    event_buffer.append(UserInputEvents([_event(ResetUserInputEventData())]))

    failures = _run(manager, event_buffer=event_buffer)

    assert failures.empty()
    assert thread.state == ["reset", "step:0:1"]


def test_close_event_ends_the_worker_without_stepping() -> None:
    event_buffer = EventBuffer()
    thread = RecordingThread(0)
    manager = _manager_for(thread)
    event_buffer.append(UserInputEvents([_event(CloseUserInputEventData())]))

    failures = _run(manager, event_buffer=event_buffer, max_steps=None)

    assert failures.empty()
    assert thread.state == []
    assert thread.latest_step is None


def test_positive_frequency_caps_the_step_rate() -> None:
    thread = RecordingThread(20)
    manager = _manager_for(thread)

    failures = _run(manager, max_steps=3)

    assert failures.empty()
    assert len(thread.step_times) == 3
    assert thread.step_times[-1] - thread.step_times[0] >= 0.09


@pytest.mark.parametrize("frequency", [-1, 1.5, True])
def test_frequency_must_be_an_unsigned_integer(frequency: int | float | bool) -> None:
    with pytest.raises((TypeError, ValueError), match="frequency"):
        RecordingThread(cast(int, frequency))
