# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical application input handling protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.core_v2.user_input_events import UserInputEvents


@runtime_checkable
class InputHandler(Protocol):
    """Provide time-windowed canonical input state to the host."""

    @abstractmethod
    def open(self) -> None:
        """Enable reading."""
        ...

    @abstractmethod
    def current_inputs(self) -> UserInputEvents:
        """Return the latest user input events."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Disable further reading."""
        ...
