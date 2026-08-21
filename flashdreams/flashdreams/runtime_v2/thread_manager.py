# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User-visible-thread ownership, communication, lifecycle, and compositing."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any, final

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.internal_thread import InternalThread

_MODEL_GENERATION_THREAD_ID = 0
"""Reserved identifier for the session's model-generation-thread."""

_THREAD_STOP_TIMEOUT_SECONDS = 30.0
"""Maximum total wait for all user-visible-threads to stop."""


class _ThreadManager:
    """Own a session's user-visible-threads and coordinate communication."""

    def __init__(self) -> None:
        self._threads: dict[int, InternalThread[Any]] = {}
        self._registry_frozen = False

    @final
    def _get_model_generation_thread_id(self) -> int:
        """Return the reserved model-generation-thread identifier."""
        return _MODEL_GENERATION_THREAD_ID

    @final
    def _register_thread(
        self,
        thread: InternalThread[Any],
        thread_id: int,
    ) -> None:
        """Register an additional user-visible-thread.

        Args:
            thread: Constructed user-visible-thread owned by this manager.
            thread_id: Positive manager-unique identifier.

        Raises:
            RuntimeError: Registration is frozen or ``thread`` already has a parent.
            TypeError: ``thread_id`` or ``thread`` has an invalid type.
            ValueError: ``thread_id`` is reserved, negative, or already registered.
        """
        if isinstance(thread_id, bool) or not isinstance(thread_id, int):
            raise TypeError("thread_id must be an integer.")
        if thread_id == _MODEL_GENERATION_THREAD_ID:
            raise ValueError("Thread ID 0 is reserved for the model-generation-thread.")
        self._register(thread, thread_id)

    @final
    def _invoke_async(
        self,
        thread_id: int,
        operation: Callable[[Any], None],
    ) -> None:
        """Send a state operation to a registered thread.

        Args:
            thread_id: Identifier of the thread that owns the state.
            operation: Callable applied before the target thread's next step.

        Raises:
            KeyError: No thread is registered under ``thread_id``.
            RuntimeError: The target thread is shutting down.
        """
        self._get_thread(thread_id)._enqueue_message(operation)

    @final
    def _register_model_generation_thread(self, thread: InternalThread[Any]) -> None:
        """Register the session's unique model-generation-thread."""
        self._register(thread, _MODEL_GENERATION_THREAD_ID)

    @final
    def _require_model_generation_thread(self) -> None:
        """Validate that session initialization registered its required thread."""
        if _MODEL_GENERATION_THREAD_ID not in self._threads:
            raise RuntimeError(
                "ISession.init() must register exactly one model-generation-thread."
            )

    def _register(self, thread: InternalThread[Any], thread_id: int) -> None:
        if self._registry_frozen:
            raise RuntimeError("Cannot register a thread after the session starts.")
        if not isinstance(thread, InternalThread):
            raise TypeError("thread must be an InternalThread instance.")
        if isinstance(thread_id, bool) or not isinstance(thread_id, int):
            raise TypeError("thread_id must be an integer.")
        if thread_id < 0:
            raise ValueError("Thread IDs must be >= 0.")
        if thread_id in self._threads:
            raise ValueError(f"Thread ID {thread_id} is already registered.")
        thread._bind_thread_manager(self)
        self._threads[thread_id] = thread

    def _get_thread(self, thread_id: int) -> InternalThread[Any]:
        try:
            return self._threads[thread_id]
        except KeyError as error:
            raise KeyError(f"No thread is registered with ID {thread_id}.") from error

    @final
    def _freeze(self) -> dict[int, InternalThread[Any]]:
        """Freeze registration and return threads in compositing order."""
        self._registry_frozen = True
        return dict(self._threads)

    @final
    def _start(
        self,
        *,
        event_buffer: EventBuffer,
        stop: threading.Event,
        failure: queue.Queue[BaseException],
        finished: threading.Event,
        max_steps: int | None,
    ) -> None:
        """Start all registered user-visible-threads."""
        for thread_id, user_visible_thread in self._freeze().items():
            is_model_generation = thread_id == _MODEL_GENERATION_THREAD_ID
            thread_name = (
                "flashdreams-model-generation-thread"
                if is_model_generation
                else f"flashdreams-user-visible-thread-{thread_id}"
            )
            user_visible_thread._start(
                thread_id=thread_id,
                thread_name=thread_name,
                event_buffer=event_buffer,
                stop=stop,
                failure=failure,
                finished=finished if is_model_generation else None,
                max_steps=max_steps if is_model_generation else None,
            )

    @final
    def _stop(self, timeout_seconds: float = _THREAD_STOP_TIMEOUT_SECONDS) -> None:
        """Stop all threads within one shared timeout.

        Args:
            timeout_seconds: Maximum total seconds to wait for every
                user-visible-thread.

        Raises:
            ValueError: ``timeout_seconds`` is negative.
            TimeoutError: One or more user-visible-threads remain alive after
                the timeout.
        """
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be >= 0.")

        threads = self._freeze()
        for user_visible_thread in threads.values():
            user_visible_thread._stop_accepting_messages()

        deadline = time.monotonic() + timeout_seconds
        timed_out: list[int] = []
        for thread_id, user_visible_thread in threads.items():
            remaining = max(0.0, deadline - time.monotonic())
            if not user_visible_thread._join(timeout=remaining):
                timed_out.append(thread_id)

        for user_visible_thread in threads.values():
            user_visible_thread._empty_message_queue()

        if timed_out:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g} seconds waiting for "
                f"user-visible-threads to stop: {timed_out}."
            )


__all__: list[str] = []
