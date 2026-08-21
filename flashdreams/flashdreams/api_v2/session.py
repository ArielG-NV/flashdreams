# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application session abstract interface."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, final

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.internal_session import InternalSession
from flashdreams.runtime_v2.session_desc import SessionDesc


class ISession(InternalSession):
    """One run of an application, and the state that run builds up.

    Created by :meth:`IApplication.create_session`. Holds the KV cache, game
    state and anything else that must not carry into another run; anything shared
    between runs belongs to the application. :meth:`init` registers one model
    model-generation-thread and any additional user-visible-threads the run needs.

    Note:
        Call order is :meth:`init`, then the registered user-visible-threads run
        until the model-generation-thread finishes. :meth:`close` runs after
        every user-visible-thread has stopped.
    """

    @final
    def register_model_generation_thread(
        self,
        thread_type: type[IThread[Any]],
        *,
        state: Any,
        **thread_kwargs: Any,
    ) -> None:
        """Construct and register the session's model-generation-thread.

        Exactly one model-generation-thread must be registered during
        :meth:`init`. It receives the reserved thread ID zero and uses the
        :class:`SessionDesc` configured model-step frequency.

        Args:
            thread_type: :class:`IThread` subclass to construct.
            state: Mutable state owned by the model-generation-thread.
            **thread_kwargs: Additional constructor arguments passed to
                ``thread_type``.

        Raises:
            RuntimeError: The session has started.
            TypeError: ``thread_type`` is not an :class:`IThread` subclass.
            ValueError: A model-generation-thread is already registered.
        """
        thread = self._construct_thread(
            thread_type,
            state=state,
            frequency=self.session_desc.frames_per_second_for_step,
            **thread_kwargs,
        )
        self._ensure_thread_manager()._register_model_generation_thread(thread)

    @final
    def get_model_generation_thread_id(self) -> int:
        """Return the ID of the model-generation-thread."""
        return self._ensure_thread_manager()._get_model_generation_thread_id()

    @final
    def register_thread(
        self,
        thread_id: int,
        thread_type: type[IThread[Any]],
        *,
        state: Any,
        frequency: int,
        **thread_kwargs: Any,
    ) -> None:
        """Construct and register one additional user-visible-thread.

        Args:
            thread_id: Positive session-unique identifier. Zero is reserved.
            thread_type: :class:`IThread` subclass to construct.
            state: Mutable state owned by the user-visible-thread.
            frequency: Maximum steps per second. Zero runs without pacing.
            **thread_kwargs: Additional constructor arguments, such as
                ``output_layout``, ``width``, and ``height`` for an ImGUIThread.

        Raises:
            RuntimeError: The session has started.
            TypeError: ``thread_type`` is not an :class:`IThread` subclass, or
                ``thread_id`` has an invalid type.
            ValueError: ``thread_id`` is reserved, negative, or already registered.
        """
        thread = self._construct_thread(
            thread_type,
            state=state,
            frequency=frequency,
            **thread_kwargs,
        )
        self._ensure_thread_manager()._register_thread(thread, thread_id)

    @final
    def invoke_async(
        self,
        thread_id: int,
        operation: Callable[[Any], None],
    ) -> None:
        """Schedule a state operation on one registered user-visible-thread.

        This may be used during :meth:`init`; the operation runs on the target
        user-visible-thread before its first step.

        Args:
            thread_id: Identifier of the user-visible-thread that owns the state.
            operation: Callable applied to the user-visible-thread-owned state.

        Raises:
            KeyError: No thread is registered under ``thread_id``.
            RuntimeError: The target user-visible-thread is shutting down.
        """
        self._ensure_thread_manager()._invoke_async(thread_id, operation)

    @abstractmethod
    def init(self) -> None:
        """Load the model and anything else this run needs.

        Must not do client I/O, since this can run before a client connects.
        Must register exactly one model-generation-thread and every additional
        user-visible-thread needed by the run.

        Messages queued through :meth:`invoke_async` here run before the target
        user-visible-thread's first call to :meth:`IThread.step`.
        """
        ...

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to create this session.

        The runtime reads it before :meth:`init` runs, since it opens the client
        window with it.
        """
        ...

    def close(self) -> None:
        """Release whatever this run holds.

        Runs even when :meth:`init` raised and after every user-visible-thread
        has stopped.
        Not abstract, and does nothing by default, so a session with nothing to
        release does not implement it.
        """
        return
