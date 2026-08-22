# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 threaded session runner."""

import gc
import inspect
import threading
import weakref
from typing import Any

import pytest
import torch
from numpy import uint64

from flashdreams import invoke_async, reserve_thread_id
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2 import thread_manager as thread_manager_module
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.internal_thread import InternalThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import (
    PresentationCordinator,
    WhenFull,
    run_session,
)
from flashdreams.runtime_v2.step_result import PresentationMode, StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

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


class SessionModelThread(IThread["FakeSession"]):
    """Run model generation against the fake session's controllable state."""

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        return self.state.step(step_index, events)

    def is_finished(self) -> bool:
        return self.state.is_finished()

    def reset(self) -> None:
        self.state.reset()


class FakeSession(ISession):
    """Produce one small RGB frame per model-generation-thread step."""

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
        main_generation_thread_id = self.register_main_generation_thread(
            SessionModelThread, state=self
        )
        self.set_layer_order_via_thread_id([main_generation_thread_id])

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self._log.record(f"session.step({step_index})")
        self.observed_events.append(events)
        if step_index == self._fail_at:
            raise RuntimeError("step failed")
        return _result(step_index, float(step_index))

    def reset(self) -> None:
        self._log.record("session.reset")

    def is_finished(self) -> bool:
        return False

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
    presentation_mode: PresentationMode = PresentationMode.SHOW_PRESENTATION,
) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.full((1, channels, 1, 2, 2), value),
        frame_count=1,
        output_layout=VideoTensorLayout.bcthw,
        presentation_mode=presentation_mode,
    )


def _event(data: UserInputEventData) -> UserInputEvents:
    return UserInputEvents([UserInputEvent(timestamp=uint64(0), event_data=data)])


def test_model_generation_runs_on_reserved_thread_and_presents_latest() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingWindow(log)

    run_session(session, window, steps=3)

    assert [call for call in log.calls if call.startswith("session.step")] == [
        "session.step(0)",
        "session.step(1)",
        "session.step(2)",
    ]
    assert log.threads_for("session.step") == {"flashdreams-model-generation-thread"}
    assert window.results
    assert torch.all(window.results[-1].output == 2.0)
    presented = (
        session.get_presentation_cordinator()
        .get_last_presented_frame(session.main_generation_thread_id)
        .get()
    )
    assert presented is not None
    assert presented.shape == (3, 2, 2)
    assert torch.all(presented == 2.0)


def test_last_presented_frame_is_none_before_compositing() -> None:
    session = FakeSession(_session_desc(), CallLog())
    session.init()
    presentation_cordinator = session.get_presentation_cordinator()
    container = presentation_cordinator.get_last_presented_frame(
        session.main_generation_thread_id
    )

    assert container.get() is None
    assert (
        presentation_cordinator.get_last_presented_frame(
            session.main_generation_thread_id
        )
        is container
    )
    assert {name for name in dir(container) if not name.startswith("_")} == {"get"}
    with pytest.raises(KeyError, match="No thread"):
        presentation_cordinator.get_last_presented_frame(99)


def test_last_presented_frame_does_not_cross_generations() -> None:
    session = FakeSession(_session_desc(), CallLog())
    session.init()
    presentation_cordinator = session.get_presentation_cordinator()
    container = presentation_cordinator.get_last_presented_frame(
        session.main_generation_thread_id
    )
    frame = torch.zeros((3, 2, 2))
    presentation_cordinator._record_last_presented_frame(
        session.main_generation_thread_id, frame
    )

    assert container.get() is frame

    presentation_cordinator._set_generation(1)

    assert container.get() is None


def test_window_calls_stay_on_the_main_program_thread() -> None:
    log = CallLog()
    main_program_thread = threading.current_thread().name

    run_session(FakeSession(_session_desc(), log), RecordingWindow(log), steps=1)

    assert log.threads_for("window.") == {main_program_thread}


def test_first_step_receives_input_collected_before_user_visible_threads_start() -> (
    None
):
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    key = KeyboardUserInputEventData(key="a", state=KeyboardInputState.PRESSED)

    run_session(session, RecordingWindow(log, [_event(key)]), steps=1)

    assert session.observed_events[0].get_events()[0].get_event_data() is key


def test_delayed_user_visible_thread_keeps_input_collected_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure startup covers event-cursor registration before event collection."""
    log = CallLog()
    allow_additional_thread_to_run = threading.Event()
    additional_thread_observed_input = threading.Event()
    observed_event_data: list[UserInputEventData] = []
    original_run = InternalThread._run
    original_collect_garbage = EventBuffer.collect_garbage
    additional_thread_id: int | None = None

    def delayed_run(user_visible_thread: InternalThread[Any], **kwargs: Any) -> None:
        if kwargs["thread_id"] == additional_thread_id:
            assert allow_additional_thread_to_run.wait(timeout=2)
        original_run(user_visible_thread, **kwargs)

    def collect_garbage(buffer: EventBuffer) -> int:
        removed = original_collect_garbage(buffer)
        allow_additional_thread_to_run.set()
        return removed

    monkeypatch.setattr(InternalThread, "_run", delayed_run)
    monkeypatch.setattr(EventBuffer, "collect_garbage", collect_garbage)

    class AdditionalThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            if events.get_events():
                observed_event_data.extend(
                    event.get_event_data() for event in events.get_events()
                )
                additional_thread_observed_input.set()
            return _result(
                step_index,
                0.0,
                presentation_mode=PresentationMode.DISABLE_PRESENTATION,
            )

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            nonlocal additional_thread_id
            super().init()
            additional_thread_id = self.register_thread(
                AdditionalThread, state=None, frequency=0
            )
            self.set_layer_order_via_thread_id(
                [self.main_generation_thread_id, additional_thread_id]
            )

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            assert additional_thread_observed_input.wait(timeout=2)
            return super().step(step_index, events)

    key = KeyboardUserInputEventData(key="a", state=KeyboardInputState.PRESSED)
    run_session(
        Session(_session_desc(), log), RecordingWindow(log, [_event(key)]), steps=1
    )

    assert additional_thread_observed_input.is_set()
    assert observed_event_data == [key]


def test_close_before_user_visible_threads_start_closes_without_a_step() -> None:
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


def test_additional_user_visible_thread_receives_async_state_operation() -> None:
    log = CallLog()
    operation_thread: list[str] = []
    operation_done = threading.Event()

    class AdditionalThread(IThread[dict[str, int]]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(
                step_index,
                0.0,
                presentation_mode=PresentationMode.DISABLE_PRESENTATION,
            )

        def reset(self) -> None:
            self.state.clear()

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            additional_thread_id = self.register_thread(
                AdditionalThread,
                state={"value": 0},
                frequency=0,
            )
            self.set_layer_order_via_thread_id(
                [self.main_generation_thread_id, additional_thread_id]
            )

            def update(state: dict[str, int]) -> None:
                operation_thread.append(threading.current_thread().name)
                state["value"] = 7
                operation_done.set()

            invoke_async(additional_thread_id, update)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            assert operation_done.wait(timeout=2)
            return super().step(step_index, events)

    session = Session(_session_desc(), log)
    run_session(session, RecordingWindow(log), steps=1)

    additional_thread_ids = set(session._ensure_thread_manager()._freeze()) - {
        session.main_generation_thread_id
    }
    assert len(additional_thread_ids) == 1
    assert operation_thread == [
        f"flashdreams-user-visible-thread-{additional_thread_ids.pop()}"
    ]


def test_session_registration_allocates_and_returns_thread_id() -> None:
    log = CallLog()

    class AdditionalThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    session = FakeSession(_session_desc(), log)
    thread_id = session.register_thread(AdditionalThread, state=None, frequency=0)
    additional_thread = session._ensure_thread_manager()._get_thread(thread_id)

    assert isinstance(thread_id, int)
    assert isinstance(additional_thread, AdditionalThread)
    invalid_thread_type: Any = object
    with pytest.raises(TypeError, match="IThread subclass"):
        session.register_thread(invalid_thread_type, state=None, frequency=0)
    with pytest.raises(TypeError, match="thread_id cannot be specified"):
        session.register_thread(
            AdditionalThread,
            state=None,
            frequency=0,
            thread_id=reserve_thread_id(),
        )
    with pytest.raises(TypeError, match="thread_id cannot be specified"):
        session.register_main_generation_thread(
            SessionModelThread,
            state=session,
            thread_id=reserve_thread_id(),
        )


def test_global_async_registry_does_not_own_registered_thread() -> None:
    class AdditionalThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    session = FakeSession(_session_desc(), CallLog())
    thread_id = session.register_thread(AdditionalThread, state=None, frequency=0)
    thread_reference = weakref.ref(
        session._ensure_thread_manager()._get_thread(thread_id)
    )

    del session
    gc.collect()

    assert thread_reference() is None
    with pytest.raises(KeyError, match="No thread"):
        invoke_async(thread_id, lambda state: None)


def test_main_generation_registration_tracks_id_and_session_frequency() -> None:
    session = FakeSession(_session_desc(), CallLog())

    thread_id = session.register_main_generation_thread(
        SessionModelThread, state=session
    )
    model_generation_thread = session._ensure_thread_manager()._get_thread(thread_id)

    assert session.main_generation_thread_id == thread_id
    assert isinstance(model_generation_thread, SessionModelThread)
    assert model_generation_thread.state is session
    assert (
        model_generation_thread.frequency
        == session.session_desc.frames_per_second_for_step
    )
    with pytest.raises(ValueError, match="already registered"):
        session.register_main_generation_thread(SessionModelThread, state=session)


def test_model_generation_registration_rejects_invalid_thread_type() -> None:
    session = FakeSession(_session_desc(), CallLog())
    invalid_thread_type: Any = object

    with pytest.raises(TypeError, match="IThread subclass"):
        session.register_main_generation_thread(invalid_thread_type, state=session)


def test_session_must_register_main_generation_thread_during_init() -> None:
    log = CallLog()

    class MissingModelSession(FakeSession):
        def init(self) -> None:
            self._log.record("session.init")

    session = MissingModelSession(_session_desc(), log)

    with pytest.raises(RuntimeError, match="must register exactly one"):
        run_session(session, RecordingWindow(log), steps=1)

    assert log.calls == ["session.init", "session.close"]


def test_session_must_set_layer_order_during_init() -> None:
    log = CallLog()

    class MissingLayerOrderSession(FakeSession):
        def init(self) -> None:
            self._log.record("session.init")
            self.register_main_generation_thread(SessionModelThread, state=self)

    session = MissingLayerOrderSession(_session_desc(), log)

    with pytest.raises(RuntimeError, match="set_layer_order_via_thread_id"):
        run_session(session, RecordingWindow(log), steps=1)

    assert log.calls == ["session.init", "session.close"]


def test_layer_order_requires_every_registered_thread_exactly_once() -> None:
    class AdditionalThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    session = FakeSession(_session_desc(), CallLog())
    main_generation_thread_id = session.register_main_generation_thread(
        SessionModelThread, state=session
    )
    additional_thread_id = session.register_thread(
        AdditionalThread, state=None, frequency=0
    )

    invalid_thread_id_list: Any = (
        main_generation_thread_id,
        additional_thread_id,
    )
    with pytest.raises(TypeError, match="list of integers"):
        session.set_layer_order_via_thread_id(invalid_thread_id_list)
    with pytest.raises(ValueError, match="duplicate"):
        session.set_layer_order_via_thread_id(
            [main_generation_thread_id, main_generation_thread_id]
        )
    with pytest.raises(ValueError, match="missing"):
        session.set_layer_order_via_thread_id([main_generation_thread_id])
    with pytest.raises(ValueError, match="unknown"):
        session.set_layer_order_via_thread_id(
            [main_generation_thread_id, additional_thread_id, reserve_thread_id()]
        )

    session.set_layer_order_via_thread_id(
        [additional_thread_id, main_generation_thread_id]
    )
    assert session._ensure_thread_manager()._get_layer_order() == (
        additional_thread_id,
        main_generation_thread_id,
    )


def test_session_registration_forwards_constructor_arguments() -> None:
    class ConfiguredThread(IThread[dict[str, int]]):
        def __init__(
            self,
            *,
            state: dict[str, int],
            frequency: int,
            output_layout: VideoTensorLayout,
            width: int,
            height: int,
            label: str,
        ) -> None:
            super().__init__(state=state, frequency=frequency)
            self.output_layout = output_layout
            self.width = width
            self.height = height
            self.label = label

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    session = FakeSession(_session_desc(), CallLog())
    state = {"value": 7}

    thread_id = session.register_thread(
        ConfiguredThread,
        state=state,
        frequency=30,
        output_layout=VideoTensorLayout.tchw,
        width=1280,
        height=720,
        label="overlay",
    )
    configured_thread = session._ensure_thread_manager()._get_thread(thread_id)

    assert isinstance(configured_thread, ConfiguredThread)
    assert configured_thread.state is state
    assert configured_thread.frequency == 30
    assert configured_thread.output_layout is VideoTensorLayout.tchw
    assert (configured_thread.width, configured_thread.height) == (1280, 720)
    assert configured_thread.label == "overlay"
    assert session._ensure_thread_manager()._freeze()[thread_id] is configured_thread


def test_thread_management_is_exposed_only_through_public_interfaces() -> None:
    assert (
        "thread_id"
        not in inspect.signature(ISession.register_main_generation_thread).parameters
    )
    assert "thread_id" not in inspect.signature(ISession.register_thread).parameters
    assert "register_main_generation_thread" in ISession.__dict__
    assert "main_generation_thread_id" in ISession.__dict__
    assert "register_thread" in ISession.__dict__
    assert "set_layer_order_via_thread_id" in ISession.__dict__
    assert "invoke_async" not in ISession.__dict__
    assert "get_presentation_cordinator" in ISession.__dict__
    assert "thread_manager" not in ISession.__dict__
    assert "invoke_async" not in IThread.__dict__
    assert "get_model_generation_thread_id" not in IThread.__dict__
    assert "get_last_presented_frame" not in IThread.__dict__
    assert not hasattr(thread_manager_module, "ThreadManager")


def test_message_operation_cannot_return_a_value() -> None:
    log = CallLog()
    operation_called = threading.Event()

    class MessageTargetThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 0.0)

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            message_target_thread_id = self.register_thread(
                MessageTargetThread,
                state=None,
                frequency=0,
            )
            self.set_layer_order_via_thread_id(
                [self.main_generation_thread_id, message_target_thread_id]
            )

            def invalid_operation(state: None) -> Any:
                del state
                operation_called.set()
                return 1

            invoke_async(message_target_thread_id, invalid_operation)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            operation_called.wait(timeout=2)
            return super().step(step_index, events)

    with pytest.raises(TypeError, match="must return None"):
        run_session(Session(_session_desc(), log), RecordingWindow(log), steps=1)


def test_explicit_layer_order_overrides_thread_id_order() -> None:
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
            self._log.record("session.init")
            overlay_thread_id = self.register_thread(Overlay, state=None, frequency=0)
            main_generation_thread_id = self.register_main_generation_thread(
                SessionModelThread, state=self
            )
            assert overlay_thread_id < main_generation_thread_id
            self.set_layer_order_via_thread_id(
                [main_generation_thread_id, overlay_thread_id]
            )

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            overlay_ready.wait(timeout=2)
            return _result(step_index, 0.0)

    window = RecordingWindow(log)
    run_session(Session(_session_desc(), log), window, steps=1)

    final = window.results[-1].output
    assert torch.allclose(final[:, 0], torch.full_like(final[:, 0], 0.5))
    assert torch.allclose(final[:, 1:], torch.zeros_like(final[:, 1:]))


def test_disabled_presentation_skips_backbuffer_and_last_frame_update() -> None:
    log = CallLog()
    disabled_presentation_ready = threading.Event()

    class DisabledPresentation(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            disabled_presentation_ready.set()
            return _result(
                step_index,
                9.0,
                presentation_mode=PresentationMode.DISABLE_PRESENTATION,
            )

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            self.disabled_presentation_thread_id = self.register_thread(
                DisabledPresentation,
                state=None,
                frequency=0,
            )
            self.set_layer_order_via_thread_id(
                [
                    self.main_generation_thread_id,
                    self.disabled_presentation_thread_id,
                ]
            )

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            disabled_presentation_ready.wait(timeout=2)
            return _result(step_index, 3.0)

    window = RecordingWindow(log)
    session = Session(_session_desc(), log)
    run_session(session, window, steps=1)

    assert torch.all(window.results[-1].output == 3.0)
    assert (
        session.get_presentation_cordinator()
        .get_last_presented_frame(session.disabled_presentation_thread_id)
        .get()
        is None
    )


def test_hidden_presentation_updates_last_frame_without_affecting_backbuffer() -> None:
    log = CallLog()
    hidden_frame_published = threading.Event()

    class HiddenPresentation(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            if step_index > 0:
                hidden_frame_published.set()
            return _result(
                step_index,
                9.0,
                presentation_mode=PresentationMode.HIDE_PRESENTATION,
            )

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            self.hidden_presentation_thread_id = self.register_thread(
                HiddenPresentation,
                state=None,
                frequency=0,
            )
            self.set_layer_order_via_thread_id(
                [
                    self.main_generation_thread_id,
                    self.hidden_presentation_thread_id,
                ]
            )

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            hidden_frame_published.wait(timeout=2)
            return _result(step_index, 3.0)

    window = RecordingWindow(log)
    session = Session(_session_desc(), log)
    run_session(session, window, steps=1)

    assert torch.all(window.results[-1].output == 3.0)
    hidden = (
        session.get_presentation_cordinator()
        .get_last_presented_frame(session.hidden_presentation_thread_id)
        .get()
    )
    assert hidden is not None
    assert torch.all(hidden == 9.0)


def test_additional_frame_is_presented_while_model_generation_is_blocked() -> None:
    log = CallLog()
    additional_frame_presented = threading.Event()

    class AdditionalFrameThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            return _result(step_index, 7.0)

        def reset(self) -> None:
            return

    class Session(FakeSession):
        def init(self) -> None:
            super().init()
            additional_frame_thread_id = self.register_thread(
                AdditionalFrameThread, state=None, frequency=0
            )
            self.set_layer_order_via_thread_id(
                [self.main_generation_thread_id, additional_frame_thread_id]
            )

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            assert additional_frame_presented.wait(timeout=2)
            return super().step(step_index, events)

    class Window(RecordingWindow):
        def write(self, result: StepResult) -> None:
            super().write(result)
            if torch.all(result.output == 7.0):
                additional_frame_presented.set()

    window = Window(log)
    run_session(Session(_session_desc(), log), window, steps=1)

    assert additional_frame_presented.is_set()
    assert torch.all(window.results[0].output == 7.0)


def test_model_only_presentation_preserves_every_frame_from_result() -> None:
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

    assert len(window.results) == 1
    result = window.results[0]
    assert result.frame_count == 4
    assert result.output[0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_blocking_backpressure_writes_the_oldest_ui_composite() -> None:
    presentation = PresentationCordinator(1, WhenFull.BLOCK)
    first = _result(0, 0.0)
    second = _result(1, 1.0)
    written: list[StepResult] = []

    assert presentation.push(first, written.append) == 0
    assert presentation.push(second, written.append) == 0
    assert written == [first]

    presentation.drain(written.append)
    assert written == [first, second]


def test_drop_oldest_backpressure_keeps_the_newest_ui_composite() -> None:
    presentation = PresentationCordinator(1, WhenFull.DROP_OLDEST)
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


def test_keyboard_interrupt_stops_user_visible_threads_and_closes_resources() -> None:
    log = CallLog()

    class InterruptingWindow(RecordingWindow):
        def get_user_input_events(self) -> UserInputEvents:
            if "window.read" in self._log.calls:
                raise KeyboardInterrupt
            return super().get_user_input_events()

    with pytest.raises(KeyboardInterrupt):
        run_session(
            FakeSession(_session_desc(frames_per_second_for_ui=1000), log),
            InterruptingWindow(log),
        )

    assert log.calls[-2:] == ["window.close", "session.close"]
    assert not any(
        thread.name.startswith("flashdreams-") for thread in threading.enumerate()
    )


def test_user_visible_thread_shutdown_has_one_bounded_timeout() -> None:
    class NeverStops:
        timeout: float | None = None

        def join(self, timeout: float | None = None) -> None:
            self.timeout = timeout

        def is_alive(self) -> bool:
            return True

    class NeverStoppingThread(IThread[None]):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            raise AssertionError("user-visible-thread must not run")

        def reset(self) -> None:
            return

    native_thread = NeverStops()
    never_stopping_thread = NeverStoppingThread(state=None, frequency=0)
    never_stopping_thread._native_thread = (  # ty: ignore[invalid-assignment]
        native_thread
    )
    manager = thread_manager_module._ThreadManager()
    thread_id = reserve_thread_id()
    manager._register_thread(never_stopping_thread, thread_id)

    with pytest.raises(TimeoutError, match=rf"user-visible-threads.*\[{thread_id}\]"):
        manager._stop(timeout_seconds=0)

    assert native_thread.timeout == 0
    with pytest.raises(RuntimeError, match="shutting down"):
        invoke_async(thread_id, lambda state: None)


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
            self.never_started_thread_id = self.register_thread(
                NeverStarted,
                state=None,
                frequency=0,
            )
            invoke_async(self.never_started_thread_id, lambda state: None)
            raise RuntimeError("init failed")

    session = FailingSession(_session_desc(), log)
    with pytest.raises(RuntimeError, match="init failed"):
        run_session(session, RecordingWindow(log), steps=1)

    never_started_thread = session._ensure_thread_manager()._get_thread(
        session.never_started_thread_id
    )
    assert isinstance(never_started_thread, NeverStarted)
    assert log.calls == ["session.init", "session.close"]
    assert never_started_thread._message_queue.empty()
    with pytest.raises(RuntimeError, match="shutting down"):
        invoke_async(session.never_started_thread_id, lambda state: None)


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
