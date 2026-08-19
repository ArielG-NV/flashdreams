# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event data protocol."""

from abc import ABC, abstractmethod


class UserInputEventData(ABC):
    """Base class for data stored in `UserInputEvent`.

    Implementations provide `type_name` and may add fields for their event data
    when implementing.
    """

    @property
    @abstractmethod
    def type_name(self) -> str:
        """Return the event type name."""
        ...

    @classmethod
    def __hash__(cls) -> int:
        """Return the hash of the concrete class name.

        The value is not stable across processes.
        """
        return hash(str(cls.__name__))

    def get_type_name(self) -> str:
        """Return the event type name."""
        return self.type_name
