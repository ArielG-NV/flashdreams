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

"""User input event API."""

from enum import Enum
from typing import Protocol


class UserInputEventData(Protocol):
    """User input event data."""
    pass

class UserInputEventDataKeyboard(UserInputEventData):
    """User input event data for keyboard."""
    pass

class UserInputEventDataMouse(UserInputEventData):
    """User input event data for mouse."""
    pass

class UserInputEventDataTouch(UserInputEventData):
    """User input event data for touch."""
    pass

class UserInputEventDataGamepad(UserInputEventData):
    """User input event data for gamepad."""
    pass

class UserInputEventDataGameWheel(UserInputEventData):
    """User input event data for game wheel."""
    pass

class UserInputEventDataGameMotion(UserInputEventData):
    """User input event data for game motion."""
    pass

class UserInputEventDataUnknown(UserInputEventData):
    """User input event data for unknown."""
    pass

class UserInputEventType(Enum):
    """User input event type."""

    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    TOUCH = "touch"
    GAMEPAD = "gamepad"
    WHEEL = "wheel"
    MOTION = "motion"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == other.value

class UserInputEvent:
    """User input event."""

    event_type: UserInputEventType = UserInputEventType.UNKNOWN
    event_data: UserInputEventData = UserInputEventDataUnknown()

    def __init__(self, event_type: UserInputEventType, event_data: UserInputEventData) -> None:
        self.event_type = event_type
        self.event_data = event_data
    
