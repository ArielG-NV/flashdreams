# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-owned session behavior."""

from abc import ABC, abstractmethod
from typing import Any, final

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.thread_manager import _ThreadManager


class InternalSession(ABC):
    """Provide runtime-owned behavior for the public session interface."""

    _thread_manager: _ThreadManager
    """Manager owned by this session, initialized on first use."""

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to create this session."""
        ...

    @final
    def _ensure_thread_manager(self) -> _ThreadManager:
        """Return this session's thread manager, creating it when needed."""
        if not hasattr(self, "_thread_manager"):
            self._thread_manager = _ThreadManager()
        return self._thread_manager

    @staticmethod
    @final
    def _construct_thread(
        thread_type: type[IThread[Any]],
        *,
        state: Any,
        frequency: int,
        **thread_kwargs: Any,
    ) -> IThread[Any]:
        if not isinstance(thread_type, type) or not issubclass(thread_type, IThread):
            raise TypeError("thread_type must be an IThread subclass.")
        return thread_type(
            state=state,
            frequency=frequency,
            **thread_kwargs,
        )


__all__ = ["InternalSession"]
