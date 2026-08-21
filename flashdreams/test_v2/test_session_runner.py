# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 threaded session runner."""

import threading
from typing import Any

import pytest
import torch
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2 import session_runner as session_runner_module
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import (
    WhenFull,
    _PresentationBuffer,
    run_session,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    KeyboardUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from numpy import uint64

pytestmark = pytest.mark.ci_cpu


class CallLog:
    """Record calls and their owning native threads."""

    def __init__(self) -> None:
        self._calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def record(self, call: str) -> None:
        with self._lock:
            self._calls.append((call, threading.current_thread().name))

    @property
    def calls(self) -> list[str]:
        with self._lock:
            return [call for call, _ in self._calls]

    def threads_for(self, prefix: str) -> set[str]:
        with self._lock:
            return {thread for call, thread in self._calls if call.startswith(prefix)}


class FakeSession(ISession):
    """Produce one small RGB frame per main-generation step."""

    def __init__(
        self,
        session_desc: SessionDesc,
        log: CallLog,
        *,
        fail_at: int | None = None,
    ) -> None:
        self._session_desc = session_desc
        self._log = log
        self._fail_at = fail_at
        self.observed_events: list[UserInputEvents] = []

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def init(self) -> None:
        self._log.record("session.init")

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self._log.record(f"session.step({step_index})")
        self.observed_events.append(events)
        if step_index == self._fail_at:
            raise RuntimeError("step failed")
        return _result(step_index, float(step_index))

    def reset(self) -> None:
        self._log.record("session.reset")

    def close(self) -> None:
        self._log.record("session.close")


class RecordingWindow(IClientWindow):
    """Report scripted input and retain every presented composite."""

    def __init__(
        self,
        log: CallLog,
        events: list[UserInputEvents] | None = None,
        *,
        fail_to_open: bool = False,
        fail_to_close: bool = False,
    ) -> None:
        self._log = log
        self._events = list(events or [])
        self._fail_to_open = fail_to_open
        self._fail_to_close = fail_to_close
        self.results: list[StepResult] = []
        self.session_desc: SessionDesc | None = None
        self._lock = threading.Lock()

    def get_user_input_events(self) -> UserInputEvents:
        self._log.record("window.read")
        with self._lock:
            if self._events:
                return self._events.pop(0)
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self._log.record("window.open")
        if self._fail_to_open:
            raise RuntimeError("open failed")
        self.session_desc = session_desc

    def write(self, result: StepResult) -> None:
        self._log.record(f"window.write({result.step_index})")
        self.results.append(result)

    def close(self) -> None:
        self._log.record("window.close")
        if self._fail_to_close:
            raise RuntimeError("close failed")


def _session_desc(*, frames_per_second_for_ui: int = 1000) -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_ui=frames_per_second_for_ui,
        frames_per_second_for_step=0,
        video_width=2,
        video_height=2,
    )


def _result(
    step_index: int,
    value: float,
    *,
    channels: int = 3,
    disabled: bool = False,
) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.full((1, channels, 1, 2, 2), value),
        frame_count=1,
        output_layout=VideoTensorLayout.bcthw,
        disabled=disabled,
    )


def _event(data: UserInputEventData) -> UserInputEvents:
    return UserInputEvents([UserInputEvent(timestamp=uint64(0), event_data=data)])


def test_main_generation_runs_on_reserved_worker_and_presents_latest() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingWindow(log)

    run_session(session, window, steps=3)

    assert [call for call in log.calls if call.startswith("session.step")] == [
        "session.step(0)",
        "session.step(1)",
        "session.step(2)",
    ]
    assert log.threads_for("session.step") == {"flashdreams-session-0"}
    assert window.results
    assert torch.all(window.results[-1].output == 2.0)


def test_window_calls_stay_on_the_io_thread() -> None:
    log = CallLog()

    run_session(FakeSession(_session_desc(), log), RecordingWindow(log), steps=1)

    assert log.threads_for("window.") == {"flashdreams-io"}


def test_first_step_receives_input_collected_before_workers_start() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    key = KeyboardUserInputEventData(key="a", pressed=True)

    run_session(session, RecordingWindow(log, [_event(key)]), steps=1)

    assert session.observed_events[0].get_events()[0].get_event_data() is key


def test_close_before_worker_start_opens_and_closes_without_a_step() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)

    run_session(
        session,
        RecordingWindow(log, [_event(CloseUserInputEventData())]),
        steps=None,
    )

    assert not any(call.startswith("session.step") for call in log.calls)
    assert log.calls[-2:] == ["window.close", "session.close"]


def test_reset_discards_in_flight_output_and_restarts_step_index() -> None:
    log = CallLog()
    reset_seen = threading.Event()

    class SlowSession(FakeSession):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            if step_index == 0 and not reset_seen.is_set():
                reset_seen.wait(timeout=2)
            return super().step(step_index, events)

    class ResetWindow(RecordingWindow):
        def get_user_input_events(self) -> UserInputEvents:
            events = super().get_user_input_events()
            if any(
                isinstance(event.get_event_data(), ResetUserInputEventData)
                for event in events.get_events()
            ):
                reset_seen.set()
            return events

    session = SlowSession(_session_desc(), log)
    window = ResetWindow(
        log,
        [UserInputEvents([]), _event(ResetUserInputEventData())],
    )

    run_session(session, window, steps=2)

    assert log.calls.count("session.step(0)") == 2
    assert "session.reset" in log.calls
    assert window.results
    assert all(result.step_index >= 0 for result in window.results)


def test_auxiliary_thread_receives_async_state_operations() -> None:
    log = CallLog()
    operation_thread: list[str] = []
    operation_done = threading.Event()

    class Auxiliary(IThread[dict[str, int]]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0, disabled=True)

        def reset(self) -> None:
            self.state.clear()

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            auxiliary = Auxiliary(state={"value": 0}, frequency=0)
            self.register_thread(auxiliary, 1)

            def update(state: dict[str, int]) -> None:
                operation_thread.append(threading.current_thread().name)
                state["value"] = 7
                operation_done.set()

            self.invoke_async(1, update)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            assert operation_done.wait(timeout=2)
            return super().step(step_index, events)

    session = Session(_session_desc(), log)
    run_session(session, RecordingWindow(log), steps=1)

    assert operation_thread == ["flashdreams-session-1"]


def test_session_registry_reserves_zero_and_rejects_duplicate_ids() -> None:
    log = CallLog()

    class Auxiliary(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    session = FakeSession(_session_desc(), log)
    worker = Auxiliary(state=None, frequency=0)
    session.register_thread(worker, 1)

    assert session.get_main_generation_thread_id() == 0
    with pytest.raises(ValueError, match="reserved"):
        session.register_thread(Auxiliary(state=None, frequency=0), 0)
    with pytest.raises(ValueError, match="already registered"):
        session.register_thread(Auxiliary(state=None, frequency=0), 1)


def test_session_declares_worker_management_as_public_api() -> None:
    assert "register_thread" in ISession.__dict__
    assert "get_main_generation_thread_id" in ISession.__dict__
    assert "invoke_async" in ISession.__dict__


def test_message_operation_cannot_return_a_value() -> None:
    log = CallLog()
    operation_called = threading.Event()

    class Auxiliary(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            self.register_thread(Auxiliary(state=None, frequency=0), 1)

            def invalid_operation(state: None) -> Any:
                del state
                operation_called.set()
                return 1

            self.invoke_async(1, invalid_operation)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            operation_called.wait(timeout=2)
            return super().step(step_index, events)

    with pytest.raises(TypeError, match="must return None"):
        run_session(Session(_session_desc(), log), RecordingWindow(log), steps=1)


def test_higher_thread_ids_alpha_composite_over_lower_ids() -> None:
    log = CallLog()
    overlay_ready = threading.Event()

    class Overlay(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            frame = _result(step_index, 0.0, channels=4)
            frame.output[:, 0] = 1.0
            frame.output[:, 3] = 0.5
            overlay_ready.set()
            return frame

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            self.register_thread(Overlay(state=None, frequency=0), 1)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            overlay_ready.wait(timeout=2)
            return _result(step_index, 0.0)

    window = RecordingWindow(log)
    run_session(Session(_session_desc(), log), window, steps=1)

    final = window.results[-1].output
    assert torch.allclose(final[:, 0], torch.full_like(final[:, 0], 0.5))
    assert torch.allclose(final[:, 1:], torch.zeros_like(final[:, 1:]))


def test_disabled_auxiliary_frame_leaves_main_frame_visible() -> None:
    log = CallLog()
    auxiliary_ready = threading.Event()

    class Disabled(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            auxiliary_ready.set()
            return _result(step_index, 9.0, disabled=True)

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            self.register_thread(Disabled(state=None, frequency=0), 1)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            auxiliary_ready.wait(timeout=2)
            return _result(step_index, 3.0)

    window = RecordingWindow(log)
    run_session(Session(_session_desc(), log), window, steps=1)

    assert torch.all(window.results[-1].output == 3.0)


def test_auxiliary_frame_is_presented_while_main_generation_is_blocked() -> None:
    log = CallLog()
    auxiliary_presented = threading.Event()

    class Auxiliary(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 7.0)

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            self.register_thread(Auxiliary(state=None, frequency=0), 1)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            assert auxiliary_presented.wait(timeout=2)
            return super().step(step_index, events)

    class Window(RecordingWindow):
        def write(self, result: StepResult) -> None:
            super().write(result)
            if torch.all(result.output == 7.0):
                auxiliary_presented.set()

    window = Window(log)
    run_session(Session(_session_desc(), log), window, steps=1)

    assert auxiliary_presented.is_set()
    assert torch.all(window.results[0].output == 7.0)


def test_compositing_selects_the_latest_frame_from_each_result() -> None:
    log = CallLog()

    class MultiFrameSession(FakeSession):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            self._log.record(f"session.step({step_index})")
            frames = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1)
            return StepResult(
                step_index=step_index,
                output=frames.expand(1, 3, 4, 2, 2),
                frame_count=4,
                output_layout=VideoTensorLayout.bcthw,
            )

    window = RecordingWindow(log)
    run_session(
        MultiFrameSession(_session_desc(), log),
        window,
        steps=1,
        max_pending=1,
        when_full=WhenFull.BLOCK,
    )

    assert window.results
    assert all(result.output[0, 0, 0, 0, 0].item() == 3.0 for result in window.results)


def test_blocking_backpressure_writes_the_oldest_ui_composite() -> None:
    presentation = _PresentationBuffer(1, WhenFull.BLOCK)
    first = _result(0, 0.0)
    second = _result(1, 1.0)
    written: list[StepResult] = []

    assert presentation.push(first, written.append) == 0
    assert presentation.push(second, written.append) == 0
    assert written == [first]

    presentation.drain(written.append)
    assert written == [first, second]


def test_drop_oldest_backpressure_keeps_the_newest_ui_composite() -> None:
    presentation = _PresentationBuffer(1, WhenFull.DROP_OLDEST)
    first = _result(0, 0.0)
    second = _result(1, 1.0)
    written: list[StepResult] = []

    assert presentation.push(first, written.append) == 0
    assert presentation.push(second, written.append) == 1
    assert written == []

    presentation.drain(written.append)
    assert written == [second]


def test_pending_frame_bound_must_be_positive() -> None:
    log = CallLog()

    with pytest.raises(ValueError, match="max_pending"):
        run_session(
            FakeSession(_session_desc(), log), RecordingWindow(log), max_pending=0
        )


def test_session_and_window_close_after_step_failure() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log, fail_at=0)

    with pytest.raises(RuntimeError, match="step failed"):
        run_session(session, RecordingWindow(log), steps=1)

    assert log.calls[-2:] == ["window.close", "session.close"]


def test_io_join_uses_bounded_waits_that_propagate_keyboard_interrupt() -> None:
    class InterruptibleThread:
        timeout: float | None = None

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.timeout = timeout
            raise KeyboardInterrupt

    thread: Any = InterruptibleThread()

    with pytest.raises(KeyboardInterrupt):
        session_runner_module._join_interruptibly(thread)

    assert thread.timeout == 0.1


def test_keyboard_interrupt_stops_workers_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = CallLog()

    def interrupt_join(thread: threading.Thread) -> None:
        assert thread.name == "flashdreams-io"
        raise KeyboardInterrupt

    monkeypatch.setattr(session_runner_module, "_join_interruptibly", interrupt_join)

    with pytest.raises(KeyboardInterrupt):
        run_session(FakeSession(_session_desc(), log), RecordingWindow(log))

    assert log.calls[-2:] == ["window.close", "session.close"]
    assert not any(
        thread.name.startswith("flashdreams-") for thread in threading.enumerate()
    )


def test_partly_initialized_session_is_closed_without_opening_window() -> None:
    log = CallLog()

    class NeverStarted(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    class FailingSession(FakeSession):
        def init(self) -> None:
            super().init()
            self.worker = NeverStarted(state=None, frequency=0)
            self.register_thread(self.worker, 1)
            self.invoke_async(1, lambda state: None)
            raise RuntimeError("init failed")

    session = FailingSession(_session_desc(), log)
    with pytest.raises(RuntimeError, match="init failed"):
        run_session(session, RecordingWindow(log), steps=1)

    assert log.calls == ["session.init", "session.close"]
    assert session.worker._message_queue.empty()
    with pytest.raises(RuntimeError, match="shutting down"):
        session.invoke_async(1, lambda state: None)


@pytest.mark.parametrize("fail_at", ["open", "close"])
def test_window_failures_are_reported_after_session_cleanup(fail_at: str) -> None:
    log = CallLog()
    window = RecordingWindow(
        log,
        fail_to_open=fail_at == "open",
        fail_to_close=fail_at == "close",
    )

    with pytest.raises(RuntimeError, match=f"{fail_at} failed"):
        run_session(FakeSession(_session_desc(), log), window, steps=1)

    assert log.calls[-1] == "session.close"
