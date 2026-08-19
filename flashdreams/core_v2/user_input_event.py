# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event API."""

from dataclasses import dataclass
from typing import ClassVar

from numpy import uint64

from flashdreams.api_v2.user_input_event_data import UserInputEventData

# These are stubbed input event data implementations for the sake of future implementation.
@dataclass(frozen=True, slots=True, eq=False)
class NumeralKeypadUserInputEventData(UserInputEventData):
    """User input event data for numeral keypad."""

    name: ClassVar[str] = "numeral_keypad"
    value: int = 0
    """The number pressed."""

# These are stubbed input event data implementations for the sake of future implementation.
@dataclass(frozen=True, slots=True, eq=False)
class KeyboardUserInputEventData(UserInputEventData):
    """User input event data for keyboard."""

    name: ClassVar[str] = "keyboard"


@dataclass(frozen=True, slots=True, eq=False)
class MouseUserInputEventData(UserInputEventData):
    """User input event data for mouse."""

    name: ClassVar[str] = "mouse"

@dataclass(frozen=True, slots=True, eq=False)
class TouchUserInputEventData(UserInputEventData):
    """User input event data for touch."""

    name: ClassVar[str] = "touch"


@dataclass(frozen=True, slots=True, eq=False)
class GamepadUserInputEventData(UserInputEventData):
    """User input event data for gamepad."""

    name: ClassVar[str] = "gamepad"


@dataclass(frozen=True, slots=True, eq=False)
class GameWheelUserInputEventData(UserInputEventData):
    """User input event data for game wheel."""

    name: ClassVar[str] = "game_wheel"


@dataclass(frozen=True, slots=True, eq=False)
class XRControllerUserInputEventData(UserInputEventData):
    """User input event data for XR controllers."""

    name: ClassVar[str] = "xr_controller"


@dataclass(frozen=True, slots=True, eq=False)
class UnknownUserInputEventData(UserInputEventData):
    """User input event data for unknown."""

    name: ClassVar[str] = "unknown"


@dataclass(frozen=True, slots=True)
class UserInputEvent:
    """User input event."""

    timestamp: uint64
    """Timestamp in microseconds since the start of the session."""

    event_data: UserInputEventData
    """Event data."""

    def get_timestamp(self) -> uint64:
        """Return the timestamp."""
        return self.timestamp

    def get_event_data(self) -> UserInputEventData:
        """Return the event data structure with type & data."""
        return self.event_data
