# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical application input handling protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.user_input_events import UserInputEvents


@runtime_checkable
class InputHandler(Protocol):
    """Reports a list of the latest collected user input events."""

    @abstractmethod
    def get_user_input_events(self) -> UserInputEvents:
        """Return all collected UserInputEvents. Implementor decides whether to empty the backing-store of inputs once called."""
        ...
