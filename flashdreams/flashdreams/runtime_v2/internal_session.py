# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-owned session behavior."""

from abc import ABC, abstractmethod
from typing import final

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.thread_manager import _ThreadManager
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class InternalSession(ABC):
    """Provide runtime-owned behavior for the public session interface."""

    _thread_manager: _ThreadManager
    """Manager owned by this session, initialized on first use."""

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to create this session."""
        ...

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Produce one result for ``step_index``."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset per-generation state."""
        ...

    @abstractmethod
    def is_finished(self) -> bool:
        """Report whether main generation should stop before its next step."""
        ...

    @final
    def _ensure_thread_manager(self) -> _ThreadManager:
        """Return this session's thread manager, creating it when needed."""
        if not hasattr(self, "_thread_manager"):
            self._thread_manager = _ThreadManager()
        return self._thread_manager

    @final
    def _register_main_generation_thread(self) -> None:
        """Register the adapter that runs :meth:`step` as the main thread."""
        manager = self._ensure_thread_manager()
        manager._register_main_thread(
            _MainGenerationThread(
                state=self,
                frequency=self.session_desc.frames_per_second_for_step,
            )
        )


class _MainGenerationThread(IThread[InternalSession]):
    """Adapt the session's generation methods to the worker contract."""

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Delegate one generation step to the session."""
        return self.state.step(step_index, events)

    def reset(self) -> None:
        """Delegate a generation reset to the session."""
        self.state.reset()

    def _is_finished(self) -> bool:
        """Let a finite session end without a client close event."""
        return self.state.is_finished()


__all__ = ["InternalSession"]
