# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session worker-thread interfaces."""

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
        """Produce one result from the events received since the previous step."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state before the first step of a new generation."""
        ...


class UIThread(IThread[StateT], ABC):
    """Wrap one-frame UI rendering in the regular worker-step contract."""

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

    @abstractmethod
    def wait_for_ui_to_render(self, frame: Tensor) -> Tensor:
        """Wait for rendering to finish and return the CUDA-visible frame."""
        ...

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Render one UI frame and wrap it in a step result."""
        frame = self.wait_for_ui_to_render(self.step_ui(step_index, events))
        return StepResult(
            step_index=step_index,
            output=frame,
            frame_count=1,
            output_layout=self.output_layout,
        )


__all__ = ["IThread", "Message", "UIThread"]
