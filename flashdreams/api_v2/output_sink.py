# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application output delivery protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.step_result import StepResult


@runtime_checkable
class OutputSink(Protocol):
    """Write to the sink while the sink is open."""

    @abstractmethod
    def open(self) -> None:
        """Prepare to receive results."""
        ...

    @abstractmethod
    def write(self, result: StepResult) -> None:
        """Consume one result.

        Called after `open` and before `close`.

        Args:
            result: Generated output for the completed step.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Finish pending writes and release resources."""
        ...
