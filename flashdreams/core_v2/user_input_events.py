# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event collection implementation."""

from dataclasses import dataclass

from flashdreams.core_v2.user_input_event import UserInputEvent
from flashdreams.core_v2.time_window import TimeWindow

@dataclass(frozen=True)
class UserInputEventsData:
    """Data for user input events."""

    window: TimeWindow
    """Time window containing the input events."""

    events: list[UserInputEvent]
    """Input events ordered by timestamp."""


class UserInputEvents:
    """Frozen collection of user input events."""

    _data: UserInputEventsData
    """Immutable event collection data."""

    def __init__(self, window: TimeWindow, events: list[UserInputEvent]) -> None:
        self._data = UserInputEventsData(
            window=window,
            events=sorted(events, key=lambda event: event.get_timestamp()),
        )

    def get_time_window(self) -> TimeWindow:
        """Return the time window for the range of events."""
        return self._data.window

    def get_events(self) -> list[UserInputEvent]:
        """Return the user input events."""
        return self._data.events
