# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application session abstract interface."""

from abc import abstractmethod
from typing import Any, final

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.internal_session import InternalSession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class ISession(InternalSession):
    """One run of an application, and the state that run builds up.

    Created by :meth:`IApplication.create_session`. Holds the KV cache, game
    state and anything else that must not carry into another run; anything shared
    between runs belongs to the application. The runtime wraps :meth:`step` in
    thread zero. Additional workers may be registered during :meth:`init`.

    Note:
        Call order is :meth:`init`, then :meth:`step` per step from index zero.
        :meth:`reset` can happen mid-run, after which the index starts again from
        zero. :meth:`close` runs after every worker has stopped.
    """

    @final
    def register_thread(self, thread: IThread[Any], thread_id: int) -> None:
        """Register one auxiliary worker.

        Args:
            thread: Constructed worker to run with this session.
            thread_id: Positive session-unique identifier. Zero is reserved.

        Raises:
            RuntimeError: The session has started or ``thread`` has a parent.
            TypeError: ``thread`` or ``thread_id`` has an invalid type.
            ValueError: ``thread_id`` is reserved, negative, or already registered.
        """
        self._ensure_thread_manager()._register_thread(thread, thread_id)

    @abstractmethod
    def init(self) -> None:
        """Load the model and anything else this run needs.

        Must not do client I/O, since this can run before a client connects.
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
        """Report whether this session has generated everything it has to.

        The main-generation worker asks before every step. A finite session,
        including a text-to-video benchmark run writing an MP4, overrides this
        instead of relying on a client close event. Interactive sessions keep
        the default and run until their client closes.

        Returns:
            Whether the run should end before taking another generation step.
        """
        return False

    def reset(self) -> None:
        """Reset per-generation state so the session can run again.

        ``run_session`` calls this when a window reports a reset event, and then
        steps from index zero again. A session that cannot start over should say
        so rather than half-reset.

        The next :meth:`step` still receives the batch the reset arrived in,
        including the events before it, so a held key stays held across the
        restart. Ignore the older events here if this session must not inherit
        them.

        Raises:
            NotImplementedError: The session does not support reuse.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reset.")

    def close(self) -> None:
        """Release whatever this run holds.

        Runs even when :meth:`init` raised, so an implementation releases what it
        managed to acquire and tolerates being called on a session that never
        finished starting. Not abstract, and does nothing by default, so a session
        with nothing to release does not implement it.
        """
        return
