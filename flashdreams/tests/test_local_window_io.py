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

"""CPU tests for shared SlangPy local-window input handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from flashdreams.demo import LocalWindowIOFactory, SessionInfo
from flashdreams.demo.local_input import SlangPyLocalInputHandler
from flashdreams.demo.local_window import SlangPyLocalWindowPresenter
from flashdreams.runtime import DRIVER_COMMAND
from flashdreams.runtime.inputs import CanonicalInputSchema, CanonicalModality

pytestmark = pytest.mark.ci_cpu


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


class _KeyboardEvent:
    def __init__(self, key: str, event_type: str) -> None:
        self.key = SimpleNamespace(name=key)
        self._event_type = event_type

    def is_key_press(self) -> bool:
        return self._event_type == "press"

    def is_key_release(self) -> bool:
        return self._event_type == "release"

    def is_key_repeat(self) -> bool:
        return self._event_type == "repeat"


def test_local_input_handler_tracks_keyboard_levels() -> None:
    clock = _Clock()
    handler = SlangPyLocalInputHandler(
        CanonicalInputSchema(modalities=(DRIVER_COMMAND,)),
        clock=clock,
    )
    handler.open(SessionInfo())

    handler.on_keyboard_event(_KeyboardEvent("w", "press"))
    pressed = handler.current_inputs()
    clock.value += 0.1
    held = handler.current_inputs()
    handler.on_keyboard_event(_KeyboardEvent("w", "release"))
    released = handler.current_inputs()

    assert pressed.values["driver_command"]["throttle"] == 1.0
    assert pressed.window.start_s == 0.0
    assert held.window.start_s == pressed.window.end_s
    assert released.window.start_s == held.window.end_s
    assert released.window.end_s > released.window.start_s
    assert held.values["driver_command"]["throttle"] == 1.0
    assert released.values["driver_command"]["throttle"] == 0.0
    assert released.metadata["canonical_sources"] == {"driver_command": "keyboard"}


def test_local_input_handler_uses_active_sdl_gamepad_axes() -> None:
    handler = SlangPyLocalInputHandler(
        CanonicalInputSchema(modalities=(DRIVER_COMMAND,))
    )
    handler.open(SessionInfo())

    handler.on_gamepad_state(
        SimpleNamespace(left_x=0.25, left_trigger=0.4, right_trigger=0.75)
    )
    inputs = handler.current_inputs()

    assert inputs.values["driver_command"] == {
        "throttle": 0.75,
        "brake": 0.4,
        "steer": -0.25,
        "stop": False,
        "reverse": False,
    }
    assert inputs.metadata["canonical_sources"] == {"driver_command": "gamepad"}


class _Presenter:
    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}
        self.pending_events: list[_KeyboardEvent] = []
        self.process_count = 0

    def set_input_callbacks(self, **callbacks: Any) -> None:
        self.callbacks = callbacks

    def process_events(self) -> None:
        self.process_count += 1
        while self.pending_events:
            self.callbacks["on_keyboard_event"](self.pending_events.pop(0))

    def close(self) -> None:
        return


def test_local_window_rebinds_cuda_context_before_native_handle_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    calls: list[object] = []
    handles = [object()]
    presenter = object.__new__(SlangPyLocalWindowPresenter)
    presenter._spy = SimpleNamespace(
        get_cuda_current_context_native_handles=lambda: (
            calls.append("handles") or handles
        )
    )
    presenter._cuda_interop_unavailable_reason = None

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: calls.append("current_device") or 2,
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: calls.append(("set_device", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: calls.append("current_stream"),
    )

    assert presenter._cuda_existing_device_handles() == handles
    assert calls == [
        "current_device",
        ("set_device", 2),
        "current_stream",
        "handles",
    ]


def test_local_window_factory_shares_presenter_with_input_handler() -> None:
    presenter = _Presenter()
    factory = LocalWindowIOFactory(presenter_factory=lambda **_kwargs: presenter)
    handler = factory.create_input_handler(
        CanonicalInputSchema(modalities=(DRIVER_COMMAND,))
    )
    output = factory.create_output_sink()
    handler.open(SessionInfo())
    output.open(SessionInfo(video_width=64, video_height=32, frames_per_second=16.0))
    presenter.pending_events.append(_KeyboardEvent("a", "press"))

    inputs = handler.current_inputs()

    assert presenter.process_count == 1
    assert inputs.values["driver_command"]["steer"] == 1.0
    output.close()
    handler.close()


def test_local_input_handler_rejects_unknown_canonical_modality() -> None:
    schema = CanonicalInputSchema(
        modalities=(
            CanonicalModality(
                name="camera_look",
                payload_fields=frozenset({"yaw", "pitch"}),
            ),
        )
    )

    with pytest.raises(ValueError, match="camera_look"):
        SlangPyLocalInputHandler(schema)
