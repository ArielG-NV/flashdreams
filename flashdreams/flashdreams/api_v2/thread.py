# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session worker-thread interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, final

from torch import Tensor

from flashdreams.runtime_v2.internal_thread import InternalThread, Message
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

StateT = TypeVar("StateT")


class IThread(InternalThread[StateT]):
    """Run stateful session work at a bounded frequency on one OS thread."""

    @final
    def invoke_async(
        self,
        thread_id: int,
        operation: Callable[[Any], None],
    ) -> None:
        """Schedule a state operation on one registered thread.

        The operation runs before the target thread's next ``step`` or
        ``step_ui``.

        Args:
            thread_id: Identifier of the thread that owns the state.
            operation: Callable applied to the thread-owned state.

        Raises:
            KeyError: No thread is registered under ``thread_id``.
            RuntimeError: This thread is unregistered or the target is shutting down.
        """
        self._get_thread_manager()._invoke_async(thread_id, operation)

    @final
    def get_model_generation_thread_id(self) -> int:
        """Return the reserved model-generation thread identifier.

        Raises:
            RuntimeError: This thread has not been registered.
        """
        return self._get_thread_manager()._get_model_generation_thread_id()

    @final
    def get_last_presented_frame(self, thread_id: int) -> Tensor | None:
        """Return a thread's most recently presented frame.

        The returned ``[C, H, W]`` tensor is shared with the producing thread and
        must be treated as read-only. ``None`` means no enabled frame from that
        thread has been presented in the current generation.

        Args:
            thread_id: Identifier of the thread whose frame is requested.

        Returns:
            Latest presented frame, or ``None`` before its first presentation.

        Raises:
            KeyError: No thread is registered under ``thread_id``.
            RuntimeError: This thread has not been registered.
        """
        return self._get_thread_manager()._get_last_presented_frame(thread_id)

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
