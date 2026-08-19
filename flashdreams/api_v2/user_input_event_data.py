# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event data protocol."""

from typing import ClassVar, Protocol

class UserInputEventData(Protocol):
    """User input event data protocol. This protocol is open-ended regarding members a user may want to add to the event data.
    Protocol exists to provide a strong-typing for event type definition."""
    name: ClassVar[str]

    @classmethod
    def __hash__(cls) -> int:
        """Return the hash of the concrete type name."""
        return hash(str(cls.__name__))

    @classmethod
    def get_name(cls) -> str:
        """Return the type name string."""
        return cls.name
