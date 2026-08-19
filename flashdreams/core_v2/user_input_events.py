# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event collection implementation."""

from dataclasses import dataclass

from flashdreams.core_v2.user_input_event import UserInputEvent


@dataclass(frozen=True)
class UserInputEvents:
    """Frozen collection of user input events."""

    _events: list[UserInputEvent]

    def get_events(self) -> list[UserInputEvent]:
        """Return the user input events."""
        return self._events
