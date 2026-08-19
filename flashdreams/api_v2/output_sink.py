# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application output delivery protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.step_result import StepResult

@runtime_checkable
class OutputSink(Protocol):
    """Writes to implementor a StepResult."""

    @abstractmethod
    def open(self) -> None:
        """Enable writing."""
        ...

    @abstractmethod
    def write(self, result: StepResult) -> None:
        """Consume one result."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Disable further writing."""
        ...
