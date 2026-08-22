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
from typing import Any, final

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.internal_thread import InternalThread
from flashdreams.runtime_v2.thread_registry import _register_thread

_THREAD_STOP_TIMEOUT_SECONDS = 30.0
"""Maximum total wait for all user-visible-threads to stop."""


class _ThreadManager:
    """Own a session's user-visible-threads and coordinate communication."""

    def __init__(self) -> None:
        self._threads: dict[int, InternalThread[Any]] = {}
        self._main_generation_thread_id: int | None = None
        self._layer_order: tuple[int, ...] | None = None
        self._registry_frozen = False

    @final
    def _get_main_generation_thread_id(self) -> int:
        """Return the registered model-generation-thread identifier."""
        if self._main_generation_thread_id is None:
            raise RuntimeError("No model-generation-thread has been registered.")
        return self._main_generation_thread_id

    @final
    def _register_thread(
        self,
        thread: InternalThread[Any],
        thread_id: int,
    ) -> None:
        """Register one user-visible-thread under a reserved identifier.

        Args:
            thread: Constructed user-visible-thread owned by this manager.
            thread_id: Process-unique identifier returned by
                :func:`flashdreams.reserve_thread_id`.

        Raises:
            RuntimeError: Registration is frozen.
            TypeError: ``thread_id`` or ``thread`` has an invalid type.
            ValueError: ``thread_id`` was not reserved or is already registered.
        """
        self._register(thread, thread_id)

    @final
    def _register_main_generation_thread(
        self,
        thread: InternalThread[Any],
        thread_id: int,
    ) -> None:
        """Register the session's unique model-generation-thread."""
        if self._main_generation_thread_id is not None:
            raise ValueError("A model-generation-thread is already registered.")
        self._register(thread, thread_id)
        self._main_generation_thread_id = thread_id

    @final
    def _require_model_generation_thread(self) -> None:
        """Validate that session initialization registered its required thread."""
        if self._main_generation_thread_id is None:
            raise RuntimeError(
                "ISession.init() must register exactly one model-generation-thread."
            )

    @final
    def _set_layer_order_via_thread_id(self, thread_id_list: list[int]) -> None:
        """Set the bottom-to-top output layer order for registered threads."""
        if self._registry_frozen:
            raise RuntimeError("Cannot set layer order after the session starts.")
        if not isinstance(thread_id_list, list):
            raise TypeError("thread_id_list must be a list of integers.")
        if any(
            isinstance(thread_id, bool) or not isinstance(thread_id, int)
            for thread_id in thread_id_list
        ):
            raise TypeError("thread_id_list must be a list of integers.")
        if len(set(thread_id_list)) != len(thread_id_list):
            raise ValueError("thread_id_list must not contain duplicate thread IDs.")
        registered_thread_ids = set(self._threads)
        layer_thread_ids = set(thread_id_list)
        if layer_thread_ids != registered_thread_ids:
            missing = sorted(registered_thread_ids - layer_thread_ids)
            unknown = sorted(layer_thread_ids - registered_thread_ids)
            raise ValueError(
                "thread_id_list must contain every registered thread ID exactly "
                f"once; missing={missing}, unknown={unknown}."
            )
        self._layer_order = tuple(thread_id_list)

    @final
    def _get_layer_order(self) -> tuple[int, ...]:
        """Return the required bottom-to-top output layer order."""
        if self._layer_order is None:
            raise RuntimeError(
                "ISession.init() must call set_layer_order_via_thread_id()."
            )
        registered_thread_ids = set(self._threads)
        if set(self._layer_order) != registered_thread_ids:
            raise RuntimeError(
                "set_layer_order_via_thread_id() must include every registered "
                "thread ID exactly once."
            )
        return self._layer_order

    def _register(self, thread: InternalThread[Any], thread_id: int) -> None:
        if self._registry_frozen:
            raise RuntimeError("Cannot register a thread after the session starts.")
        if not isinstance(thread, InternalThread):
            raise TypeError("thread must be an InternalThread instance.")
        if isinstance(thread_id, bool) or not isinstance(thread_id, int):
            raise TypeError("thread_id must be an integer.")
        if thread_id in self._threads:
            raise ValueError(f"Thread ID {thread_id} is already registered.")
        _register_thread(thread_id, thread)
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
        self._get_layer_order()
        main_generation_thread_id = self._get_main_generation_thread_id()
        for thread_id, user_visible_thread in self._freeze().items():
            is_model_generation = thread_id == main_generation_thread_id
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
