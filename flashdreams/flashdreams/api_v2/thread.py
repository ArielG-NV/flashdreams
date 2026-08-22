# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session user-visible-thread interfaces."""

from abc import ABC, abstractmethod
from typing import TypeVar, final

from torch import Tensor

from flashdreams.runtime_v2.internal_thread import InternalThread, Message
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

StateT = TypeVar("StateT")


class IThread(InternalThread[StateT]):
    """Run stateful session work at a bounded frequency on one OS thread."""

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Produce one result for ``step_index``.

        Args:
            step_index: Zero-based index of the step to produce.
            events: User input events collected since the previous step.

        Returns:
            Result carrying ``step_index``.
        """
        ...

    def is_finished(self) -> bool:
        """Report whether this user-visible-thread has completed its workload.

        The runtime asks before every step.
        If finished, that particular thread ends.
        Additionally, if finishing the model-generation-thread, the session ends (all threads will finish after finishing current work).
        Returns:
            Whether this user-visible-thread should stop before its next step.
        """
        return False

    def reset(self) -> None:
        """Reset per-generation state so the session can run again.

        ``run_session`` calls this when a window reports a reset event, and then
        steps from index zero again.

        The next :meth:`step` still receives the batch the reset arrived in,
        including the events before it, so a held key stays held across the
        restart. Ignore the older events here if this session must not inherit
        them.

        Raises:
            NotImplementedError: The session does not support reuse.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reset.")


class UIThread(IThread[StateT], ABC):
    """Wrap an IThread to produce a one-frame-at-a-time render, with intention of use for UI."""

    def __init__(
        self,
        *,
        state: StateT,
        frequency: int,
        output_layout: VideoTensorLayout,
    ) -> None:
        super().__init__(state=state, frequency=frequency)
        self.output_layout = output_layout

    @abstractmethod
    def step_ui(self, step_index: int, events: UserInputEvents) -> Tensor:
        """Render and return one frame backed by CUDA memory."""
        ...

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Render one UI frame and wrap it in a step result."""
        frame = self.step_ui(step_index, events)
        return StepResult(
            step_index=step_index,
            output=frame,
            frame_count=1,
            output_layout=self.output_layout,
        )


__all__ = ["IThread", "Message", "UIThread"]
