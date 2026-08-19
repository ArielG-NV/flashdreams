# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event API."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from numpy import uint64


class UserInputEventData(Protocol):
    """User input event data."""
    pass

class KeyboardUserInputEventData(UserInputEventData):
    """User input event data for keyboard."""
    pass

class MouseUserInputEventData(UserInputEventData):
    """User input event data for mouse."""
    pass

class TouchUserInputEventData(UserInputEventData):
    """User input event data for touch."""
    pass

class GamepadUserInputEventData(UserInputEventData):
    """User input event data for gamepad."""
    pass

class GameWheelUserInputEventData(UserInputEventData):
    """User input event data for game wheel."""
    pass

class XRControllerUserInputEventData(UserInputEventData):
    """User input event data for XR controllers."""
    pass

@dataclass(frozen=True, slots=True)
class UnknownUserInputEventData(UserInputEventData):
    """User input event data for unknown."""

    data: object
    """Opaque application-defined input data."""

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

@dataclass(frozen=True, slots=True)
class UserInputEvent:
    """User input event."""

    timestamp: uint64
    """Timestamp in microseconds since the start of the session."""

    event_type: UserInputEventType
    """Device-independent category of the input event."""

    event_data: UserInputEventData
    """Event payload."""

    def get_timestamp(self) -> uint64:
        """Return the timestamp."""
        return self.timestamp

    def get_event_type(self) -> UserInputEventType:
        """Return the event type."""
        return self.event_type

    def get_event_data(self) -> UserInputEventData:
        """Return the event data."""
        return self.event_data
