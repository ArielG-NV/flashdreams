# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for C window callbacks and invocation-local launch behavior."""

import ctypes as ct
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from numpy import uint64

from flashdreams.runtime_v2 import c_client_window, cli, client_window_factory
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _callbacks(**overrides: Callable) -> c_client_window._ClientWindowCallbacks:
    """Keep ctypes function pointers alive through their owning structure."""
    functions = {
        "open": lambda *args: 0,
        "get_user_input_events": lambda *args: 0,
        "write": lambda *args: 0,
        "close": lambda *args: None,
        **overrides,
    }
    callbacks = c_client_window._ClientWindowCallbacks()
    callbacks.struct_size = ct.sizeof(callbacks)
    callbacks.abi_version = 2
    callbacks.user_data = 123
    for name, function in functions.items():
        setattr(callbacks, name, type(getattr(callbacks, name))(function))
    return callbacks


def test_input_is_consumed_once_and_sorted() -> None:
    events = [
        c_client_window._InputEvent(type=3, timestamp_us=20, key=b"r", pressed=1),
        c_client_window._InputEvent(type=1, timestamp_us=10),
    ]

    def poll(context, event):
        assert context == 123
        if not events:
            return 0
        event[0] = events.pop(0)
        return 1

    window = c_client_window.CClientWindow(_callbacks(get_user_input_events=poll))
    window.open(SessionDesc())
    received = window.get_user_input_events().get_events()
    assert isinstance(received[0], CloseUserInputEvent)
    assert isinstance(received[1], KeyboardUserInputEvent)
    assert received[1].key == "r"
    assert received[1].state is KeyboardInputState.PRESSED
    assert window.get_user_input_events().get_events() == []
    window.close()


def test_cuda_descriptor_context_and_cleanup() -> None:
    calls = []

    def write(context, pointer):
        frame = pointer.contents
        calls.append(
            (
                context,
                frame.device_pointer,
                frame.byte_count,
                frame.width,
                frame.height,
                frame.frames,
                frame.step_index,
                frame.device_index,
                frame.stream,
            )
        )
        return 0

    callbacks = _callbacks(
        write=write, close=lambda context: calls.append(("close", context))
    )
    window = c_client_window.CClientWindow(callbacks)
    window.open(SessionDesc(video_width=1, video_height=1))
    # No CPU materialization methods exist on this fake CUDA allocation.
    frames = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda:3"),
        shape=(1, 1, 1, 3),
        data_ptr=lambda: 0x1000,
        numel=lambda: 3,
        get_device=lambda: 3,
        record_stream=Mock(),
    )
    stream = SimpleNamespace(cuda_stream=0x2000)
    result = StepResult(
        step_index=7,
        output=torch.zeros(1, 3, 1, 1),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
    )
    with (
        patch.object(c_client_window, "result_to_rgb24_tensor", return_value=frames),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "current_stream", return_value=stream),
    ):
        window.write(result)
        callbacks.write = type(callbacks.write)(lambda *args: -1)
        with pytest.raises(RuntimeError, match="write callback failed"):
            window.write(result)
    assert frames.record_stream.call_count == 2
    frames.record_stream.assert_called_with(stream)
    window.close()
    window.close()
    assert calls == [(123, 0x1000, 3, 1, 1, 1, 7, 3, 0x2000), ("close", 123)]


def test_optional_window_requests() -> None:
    calls = []
    window = c_client_window.CClientWindow(
        _callbacks(
            request_hide_cursor=lambda *args: calls.append(("hide", *args)),
            request_lock_cursor_to_window=lambda *args: calls.append(("lock", *args)),
            request_new_window_size=lambda *args: calls.append(
                ("request_new_window_size", *args)
            ),
        )
    )
    window.request_hide_cursor(True)
    window.request_lock_cursor_to_window(False)
    window.request_new_window_size((8, 6))
    assert calls == [
        ("hide", 123, 1),
        ("lock", 123, 0),
        ("request_new_window_size", 123, 8, 6),
    ]


@pytest.mark.parametrize(
    "field", ["struct_size", "abi_version", "get_user_input_events"]
)
def test_invalid_callback_tables(field: str) -> None:
    callbacks = _callbacks()
    setattr(
        callbacks,
        field,
        0
        if field != "get_user_input_events"
        else type(callbacks.get_user_input_events)(),
    )
    with pytest.raises(ValueError):
        c_client_window.CClientWindow(callbacks)


def test_callback_failure_closes_partial_open() -> None:
    closed = []
    window = c_client_window.CClientWindow(
        _callbacks(open=lambda *args: -1, close=lambda *args: closed.append(True))
    )
    with pytest.raises(RuntimeError, match="open callback failed"):
        window.open(SessionDesc())
    window.close()
    window.close()
    assert closed == [True]
    window = c_client_window.CClientWindow(
        _callbacks(get_user_input_events=lambda *args: -1)
    )
    with pytest.raises(RuntimeError, match="get_user_input_events callback failed"):
        window.get_user_input_events()


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (c_client_window._InputEvent(type=2), ResetUserInputEvent),
        (c_client_window._InputEvent(type=4, x=2, y=3), MouseUserInputEvent),
        (c_client_window._InputEvent(type=5, button=1, pressed=1), MouseUserInputEvent),
        (c_client_window._InputEvent(type=6, x=-1, y=2), MouseUserInputEvent),
        (c_client_window._InputEvent(type=7, pressed=1), FocusUserInputEvent),
    ],
)
def test_input_types(event, expected) -> None:
    assert isinstance(c_client_window._input_event(event, uint64(1)), expected)


@pytest.mark.parametrize(
    "event",
    [
        c_client_window._InputEvent(type=99),
        c_client_window._InputEvent(type=3),
        c_client_window._InputEvent(type=3, key=b"\xff"),
        c_client_window._InputEvent(type=3, key=b"r", pressed=2),
        c_client_window._InputEvent(type=4, x=float("nan")),
    ],
)
def test_invalid_input(event) -> None:
    with pytest.raises(ValueError):
        c_client_window._input_event(event, uint64(1))


def test_launch_callbacks_are_local() -> None:
    callbacks = _callbacks()

    def launch(arguments):
        parsed = cli._parser().parse_args(arguments)
        window = client_window_factory.create_client_window(parsed)
        assert isinstance(window, c_client_window.CClientWindow)
        assert ct.addressof(window._callbacks) == ct.addressof(callbacks)

    with patch.object(c_client_window, "entrypoint", side_effect=launch):
        c_client_window._run(["app", "--mode", "c"], ct.addressof(callbacks))
    assert c_client_window._callbacks is None
    parsed = cli._parser().parse_args(["app", "--mode", "c"])
    with pytest.raises(ValueError, match="requires a C host callback table"):
        client_window_factory.create_client_window(parsed)


def test_exit_and_reentrant_launch_return_errors() -> None:
    with patch.object(c_client_window, "entrypoint", side_effect=SystemExit(0)):
        c_client_window._run(["--help"], 0)
    with patch.object(c_client_window, "entrypoint", side_effect=SystemExit(2)):
        with pytest.raises(RuntimeError, match="status 2"):
            callbacks = _callbacks()
            c_client_window._run([], ct.addressof(callbacks))
    assert c_client_window._callbacks is None
    with c_client_window._RUN_LOCK:
        with pytest.raises(RuntimeError, match="already running"):
            c_client_window._run([], 0)


@pytest.mark.parametrize("mode", ["c", "null"])
def test_runner_process_ownership_follows_selected_window(mode: str) -> None:
    application = SimpleNamespace(session_desc=lambda: SessionDesc())
    callbacks = _callbacks()
    with (
        patch.object(cli, "create_application", return_value=application),
        patch.object(cli.ApplicationRunner, "run"),
        patch.object(cli, "shutdown_distributed") as shutdown,
    ):
        c_client_window._run(["app", "--mode", mode], ct.addressof(callbacks))
    shutdown.assert_called_once_with(synchronize=True, terminate_process=mode != "c")
