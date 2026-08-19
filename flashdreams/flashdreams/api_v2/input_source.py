# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application input handling protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.user_input_events import UserInputEvents


@runtime_checkable
class InputSource(Protocol):
    """Provide user input.

    The caller calls `get_user_input_events` when it needs to see the latest list of user input events.
    That have ocurred.
    """

    @abstractmethod
    def get_user_input_events(self) -> UserInputEvents:
        """Return the user input events collected so far.

        Implementations decide whether returned events remain or empty between calls.
        """
        ...
