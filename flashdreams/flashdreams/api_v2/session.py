# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application session abstract interface."""

from abc import abstractmethod
from typing import Any, final

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.internal_session import InternalSession
from flashdreams.runtime_v2.presentation_cordinator import PresentationCordinator
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.thread_registry import reserve_thread_id


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
    def get_presentation_cordinator(self) -> PresentationCordinator:
        """Return the presentation coordinator owned by this session."""
        return self._ensure_presentation_cordinator()

    @final
    def register_main_generation_thread(
        self,
        thread_type: type[IThread[Any]],
        *,
        state: Any,
        **thread_kwargs: Any,
    ) -> int:
        """Construct and register the session's model-generation-thread.

        Exactly one model-generation-thread must be registered during
        :meth:`init`. Its frequency comes from :class:`SessionDesc`.

        Args:
            thread_type: :class:`IThread` subclass to construct.
            state: Mutable state owned by the model-generation-thread.
            **thread_kwargs: Additional constructor arguments passed to
                ``thread_type``.

        Returns:
            Process-unique identifier assigned to the model-generation-thread.

        Raises:
            RuntimeError: The session has started.
            TypeError: ``thread_type`` is not an :class:`IThread` subclass, or
                the caller supplies ``thread_id``.
            ValueError: A model-generation-thread is already registered.
        """
        self._reject_thread_id_argument(thread_kwargs)
        thread = self._construct_thread(
            thread_type,
            state=state,
            frequency=self.session_desc.frames_per_second_for_step,
            **thread_kwargs,
        )
        registered_thread_id = reserve_thread_id()
        self._ensure_thread_manager()._register_main_generation_thread(
            thread, registered_thread_id
        )
        self.get_presentation_cordinator()._register_thread(registered_thread_id)
        self._main_generation_thread_id = registered_thread_id
        return registered_thread_id

    @property
    @final
    def main_generation_thread_id(self) -> int:
        """Return the ID assigned to this session's model-generation-thread."""
        if not hasattr(self, "_main_generation_thread_id"):
            raise RuntimeError("No model-generation-thread has been registered.")
        return self._main_generation_thread_id

    @final
    def register_thread(
        self,
        thread_type: type[IThread[Any]],
        *,
        state: Any,
        frequency: int,
        **thread_kwargs: Any,
    ) -> int:
        """Construct and register one additional user-visible-thread.

        Args:
            thread_type: :class:`IThread` subclass to construct.
            state: Mutable state owned by the user-visible-thread.
            frequency: Maximum steps per second. Zero runs without pacing.
            **thread_kwargs: Additional constructor arguments, such as
                ``output_layout``, ``width``, and ``height`` for an ImGUIThread.

        Returns:
            Process-unique identifier assigned to the user-visible-thread.

        Raises:
            RuntimeError: The session has started.
            TypeError: ``thread_type`` is not an :class:`IThread` subclass, or
                the caller supplies ``thread_id``.
        """
        self._reject_thread_id_argument(thread_kwargs)
        thread = self._construct_thread(
            thread_type,
            state=state,
            frequency=frequency,
            **thread_kwargs,
        )
        registered_thread_id = reserve_thread_id()
        self._ensure_thread_manager()._register_thread(thread, registered_thread_id)
        self.get_presentation_cordinator()._register_thread(registered_thread_id)
        return registered_thread_id

    @final
    def set_layer_order_via_thread_id(self, thread_id_list: list[int]) -> None:
        """Set the mandatory bottom-to-top output layer order.

        Args:
            thread_id_list: Every registered thread ID exactly once. Index zero
                is the bottom layer and the final index is the top layer.

        Raises:
            RuntimeError: The session has started.
            TypeError: ``thread_id_list`` is not a list of integers.
            ValueError: IDs are duplicated, missing, or not registered.
        """
        self._ensure_thread_manager()._set_layer_order_via_thread_id(thread_id_list)

    @staticmethod
    def _reject_thread_id_argument(thread_kwargs: dict[str, Any]) -> None:
        if "thread_id" in thread_kwargs:
            raise TypeError(
                "thread_id cannot be specified; registration assigns it automatically."
            )

    @abstractmethod
    def init(self) -> None:
        """Load the model and anything else this run needs.

        Must not do client I/O, since this can run before a client connects.
        Must register exactly one model-generation-thread and every additional
        user-visible-thread needed by the run, then call
        :meth:`set_layer_order_via_thread_id` with every registered ID.

        Messages queued through :func:`flashdreams.invoke_async` here run before
        the target user-visible-thread's first call to :meth:`IThread.step`.
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
