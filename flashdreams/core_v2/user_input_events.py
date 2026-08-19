# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event collection implementation."""

from dataclasses import dataclass

from flashdreams.core_v2.user_input_event import UserInputEvent

@dataclass(frozen=True)
class UserInputEventsData:
    """Data for user input events."""

    events: list[UserInputEvent]
    """Input events ordered by timestamp."""


class UserInputEvents:
    """Frozen collection of user input events."""

    _data: UserInputEventsData
    """Immutable event collection data."""

    def __init__(self, events: list[UserInputEvent]) -> None:
        self._data = UserInputEventsData(
            events=sorted(events, key=lambda event: event.get_timestamp()),
        )

    def get_events(self) -> list[UserInputEvent]:
        """Return the user input events."""
        return list(self._data.events)
