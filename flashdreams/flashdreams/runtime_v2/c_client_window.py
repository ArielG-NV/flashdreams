# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedded C launch bridge and callback-backed client window."""

import ctypes as ct
import math
import os
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext

import torch
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.cli import entrypoint
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
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor


class _InputEvent(ct.Structure):
    """Native layout of ``fd_input_event`` in ``c-api/include/flashdreams.h``."""

    _fields_ = [
        ("type", ct.c_int32),
        ("timestamp_us", ct.c_uint64),
        ("key", ct.c_char_p),
        ("x", ct.c_double),
        ("y", ct.c_double),
        ("button", ct.c_int32),
        ("pressed", ct.c_int32),
    ]


class _CudaFrame(ct.Structure):
    """Native layout of the borrowed CUDA buffer descriptor ``fd_cuda_frame``."""

    _fields_ = [
        ("device_pointer", ct.c_void_p),
        ("byte_count", ct.c_size_t),
        ("width", ct.c_int),
        ("height", ct.c_int),
        ("frames", ct.c_int),
        ("step_index", ct.c_int64),
        ("device_index", ct.c_int),
        ("stream", ct.c_void_p),
    ]


class _ClientWindowCallbacks(ct.Structure):
    """Native layout of ``fd_client_window``; callbacks use the C convention."""

    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("abi_version", ct.c_uint32),
        ("user_data", ct.c_void_p),
        ("open", ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int)),
        (
            "get_user_input_events",
            ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(_InputEvent)),
        ),
        (
            "write",
            ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(_CudaFrame)),
        ),
        ("close", ct.CFUNCTYPE(None, ct.c_void_p)),
        ("request_hide_cursor", ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int)),
        ("request_lock_cursor_to_window", ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int)),
        (
            "request_new_window_size",
            ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int, ct.c_int),
        ),
    ]


class CClientWindow(IClientWindow):
    """Forward window operations to a C host for one embedded invocation.

    The host owns the callback table and its context until launch returns.
    The C launcher supplies the callbacks for ``--mode c``.
    """

    def __init__(self, callbacks: _ClientWindowCallbacks) -> None:
        if callbacks.struct_size != ct.sizeof(_ClientWindowCallbacks):
            raise ValueError("Invalid C client-window struct size.")
        if callbacks.abi_version != 2:
            raise ValueError("Unsupported C client-window ABI version.")
        if not all(
            (
                callbacks.open,
                callbacks.get_user_input_events,
                callbacks.write,
                callbacks.close,
            )
        ):
            raise ValueError(
                "CClientWindow requires open, get_user_input_events, write and close."
            )
        self._callbacks = callbacks
        self._session_desc: SessionDesc | None = None
        self._size = (0, 0)
        self._opened = False
        self._start_ns = 0

    def open(self, session_desc: SessionDesc) -> None:
        """Open the host window using the resolved session dimensions."""
        self.close()
        self._session_desc = session_desc
        self._size = (session_desc.video_width, session_desc.video_height)
        self._start_ns = time.monotonic_ns()
        self._opened = True
        if self._callbacks.open(
            self._callbacks.user_data,
            *self._size,
            session_desc.frames_per_second_for_step,
        ):
            raise RuntimeError("C client-window open callback failed.")

    def get_user_input_events(self) -> UserInputEvents:
        """Drain host input, copying borrowed key strings before polling again."""
        events: list[UserInputEvent] = []
        # ponytail: cap input work at 4096 events per UI tick; use a bulk ABI
        # if hosts need higher throughput without starving presentation.
        for _ in range(4096):
            event = _InputEvent()
            status = self._callbacks.get_user_input_events(
                self._callbacks.user_data, ct.byref(event)
            )
            if status == 0:
                break
            if status != 1:
                raise RuntimeError(
                    "C client-window get_user_input_events callback failed."
                )
            timestamp = uint64(
                event.timestamp_us or (time.monotonic_ns() - self._start_ns) // 1000
            )
            events.append(_input_event(event, timestamp))
        return UserInputEvents(events)

    def write(self, result: StepResult) -> None:
        """Deliver CUDA RGB24 pixels with stream-ordered production and lifetime."""
        if self._session_desc is None or not self._opened:
            raise RuntimeError("C client window is not open.")
        frames = result_to_rgb24_tensor(result, self._session_desc, self._size)
        if not frames.is_cuda:
            frames = frames.to(
                device=torch.device("cuda", torch.cuda.current_device()),
                non_blocking=True,
            )
        with torch.cuda.device(frames.device):
            stream = torch.cuda.current_stream(frames.device)
            frame = _CudaFrame(
                frames.data_ptr(),
                frames.numel(),
                frames.shape[2],
                frames.shape[1],
                frames.shape[0],
                result.step_index,
                frames.get_device(),
                stream.cuda_stream,
            )
            try:
                if self._callbacks.write(self._callbacks.user_data, ct.byref(frame)):
                    raise RuntimeError("C client-window write callback failed.")
            finally:
                # Host work queued during the callback must finish before the
                # allocator can recycle this storage, including callback failures.
                frames.record_stream(stream)

    def close(self) -> None:
        """Release the host window once per open, including a failed open."""
        if self._opened:
            self._opened = False
            self._callbacks.close(self._callbacks.user_data)
        self._session_desc = None

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Forward cursor visibility when the host supports it."""
        if self._callbacks.request_hide_cursor:
            self._callbacks.request_hide_cursor(self._callbacks.user_data, hide_cursor)

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Forward pointer capture when the host supports it."""
        if self._callbacks.request_lock_cursor_to_window:
            self._callbacks.request_lock_cursor_to_window(
                self._callbacks.user_data, lock_cursor_to_window
            )

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Resize host presentation when the host supports resizing."""
        if self._callbacks.request_new_window_size:
            if min(new_window_size) <= 0:
                raise ValueError("C client-window dimensions must be positive.")
            self._callbacks.request_new_window_size(
                self._callbacks.user_data, *new_window_size
            )
            self._size = new_window_size


def _input_event(event: _InputEvent, timestamp: uint64) -> UserInputEvent:
    """Translate and validate one C input event."""
    if event.type == 1:
        return CloseUserInputEvent(timestamp=timestamp)
    if event.type == 2:
        return ResetUserInputEvent(timestamp=timestamp)
    if event.pressed not in (0, 1):
        raise ValueError("C input pressed must be 0 or 1.")
    if event.type == 3:
        if not event.key:
            raise ValueError("C keyboard input requires a nonempty UTF-8 key.")
        return KeyboardUserInputEvent(
            timestamp=timestamp,
            key=event.key.decode("utf-8"),
            state=(
                KeyboardInputState.PRESSED
                if event.pressed
                else KeyboardInputState.RELEASED
            ),
        )
    if event.type in (4, 5, 6):
        if not math.isfinite(event.x) or not math.isfinite(event.y):
            raise ValueError("C mouse coordinates must be finite.")
        if event.type == 6:
            return MouseUserInputEvent(
                timestamp=timestamp, action="wheel", wheel_x=event.x, wheel_y=event.y
            )
        return MouseUserInputEvent(
            timestamp=timestamp,
            action="move" if event.type == 4 else "button",
            x=event.x,
            y=event.y,
            button=event.button,
            pressed=bool(event.pressed),
        )
    if event.type == 7:
        return FocusUserInputEvent(timestamp=timestamp, focused=bool(event.pressed))
    raise ValueError(f"Unknown C input event type: {event.type}.")


_RUN_LOCK = threading.Lock()
"""Reject overlapping and callback-reentrant launches in the interpreter."""

_callbacks: _ClientWindowCallbacks | None = None
"""Host callbacks available until the current C launch returns."""


def _run(arguments: Sequence[str], callbacks_address: int) -> None:
    """Launch from ``fd_run`` with the host's callback table."""
    global _callbacks
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("A C API application launch is already running.")
    try:
        if callbacks_address:
            callbacks = _ClientWindowCallbacks.from_address(callbacks_address)
            # Validate even if another mode is selected, before application init.
            CClientWindow(callbacks)
            _callbacks = callbacks
        try:
            # Stable-ABI extensions need python3.dll from the base interpreter
            # directory, which is not the executable directory in a C host.
            with (
                os.add_dll_directory(sys.base_prefix)
                if sys.platform == "win32"
                else nullcontext()
            ):
                entrypoint(arguments)
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError(
                    f"Application exited with status {error.code}."
                ) from error
    finally:
        _callbacks = None
        _RUN_LOCK.release()
