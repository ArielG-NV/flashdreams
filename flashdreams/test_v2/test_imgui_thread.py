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

"""CPU tests for Dear ImGui-thread input routing."""

from types import SimpleNamespace
from typing import Any

import pytest
from numpy import uint64

from flashdreams.runtime_v2.imgui_thread import (
    SlangPyImGUIRenderer,
    _route_input_events,
)
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _IO:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def add_mouse_pos_event(self, x: float, y: float) -> None:
        self.calls.append(("position", x, y))

    def add_mouse_button_event(self, button: int, pressed: bool) -> None:
        self.calls.append(("button", button, pressed))

    def add_mouse_wheel_event(self, x: float, y: float) -> None:
        self.calls.append(("wheel", x, y))

    def add_key_event(self, key: object, pressed: bool) -> None:
        self.calls.append(("key", key, pressed))

    def add_input_characters_utf8(self, text: str) -> None:
        self.calls.append(("text", text))

    def add_focus_event(self, focused: bool) -> None:
        self.calls.append(("focus", focused))


def _event(index: int, data: Any) -> UserInputEvent:
    return UserInputEvent(timestamp=uint64(index), event_data=data)


def test_routes_all_imgui_input_categories_in_order() -> None:
    io = _IO()
    imgui = SimpleNamespace(Key=SimpleNamespace(left_arrow="left"))
    events = UserInputEvents(
        [
            _event(1, MouseUserInputEventData(action="move", x=0.25, y=0.5)),
            _event(
                2,
                MouseUserInputEventData(
                    action="button", x=0.25, y=0.5, button=2, pressed=True
                ),
            ),
            _event(
                3,
                MouseUserInputEventData(
                    action="wheel", x=0.25, y=0.5, wheel_x=-1.0, wheel_y=2.0
                ),
            ),
            _event(
                4,
                KeyboardUserInputEventData(
                    key="ArrowLeft", state=KeyboardInputState.PRESSED
                ),
            ),
            _event(
                5,
                KeyboardUserInputEventData(key="h", state=KeyboardInputState.PRESSED),
            ),
            _event(
                6,
                KeyboardUserInputEventData(
                    key="ArrowLeft", state=KeyboardInputState.RELEASED
                ),
            ),
            _event(7, FocusUserInputEventData(focused=True)),
        ]
    )

    _route_input_events(events, io=io, imgui=imgui, width=640, height=480)

    assert io.calls == [
        ("position", 160.0, 240.0),
        ("position", 160.0, 240.0),
        ("button", 2, True),
        ("position", 160.0, 240.0),
        ("wheel", -1.0, 2.0),
        ("key", "left", True),
        ("text", "h"),
        ("key", "left", False),
        ("focus", True),
    ]


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 1)])
def test_slangpy_renderer_rejects_nonpositive_dimensions(
    width: int, height: int
) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        SlangPyImGUIRenderer(width=width, height=height)
