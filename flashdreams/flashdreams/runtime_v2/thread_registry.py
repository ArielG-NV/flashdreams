# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide user-visible-thread identifiers and weak lookup."""

from __future__ import annotations

import itertools
import threading
import weakref
from collections.abc import Callable
from typing import Any, Protocol, TypeVar


class _MessageTarget(Protocol):
    """Minimal interface needed by the process-wide message registry."""

    def _enqueue_message(self, operation: Callable[[Any], None]) -> None: ...


StateT = TypeVar("StateT")

_thread_ids = itertools.count()
_reserved_thread_ids: set[int] = set()
_threads: dict[int, weakref.ReferenceType[_MessageTarget]] = {}
_registry_lock = threading.Lock()


def reserve_thread_id() -> int:
    """Reserve and return a process-unique user-visible-thread identifier."""
    with _registry_lock:
        thread_id = next(_thread_ids)
        _reserved_thread_ids.add(thread_id)
        return thread_id


def invoke_async(
    thread_id: int,
    operation: Callable[[StateT], None],
) -> None:
    """Schedule ``operation`` on the user-visible-thread identified by ``thread_id``.

    The registry holds only a weak reference. The target remains alive only while
    its session owns it, and the operation runs before that target's next step.

    Raises:
        TypeError: ``thread_id`` is not an integer.
        KeyError: No live user-visible-thread is registered under ``thread_id``.
        RuntimeError: The target user-visible-thread is shutting down.
    """
    if isinstance(thread_id, bool) or not isinstance(thread_id, int):
        raise TypeError("thread_id must be an integer.")
    with _registry_lock:
        thread_reference = _threads.get(thread_id)
        thread = thread_reference() if thread_reference is not None else None
    if thread is None:
        raise KeyError(f"No thread is registered with ID {thread_id}.")
    thread._enqueue_message(operation)


def _register_thread(thread_id: int, thread: _MessageTarget) -> None:
    """Associate a reserved identifier with ``thread`` without owning it."""
    if isinstance(thread_id, bool) or not isinstance(thread_id, int):
        raise TypeError("thread_id must be an integer.")
    with _registry_lock:
        if thread_id in _threads:
            raise ValueError(f"Thread ID {thread_id} is already registered.")
        if thread_id not in _reserved_thread_ids:
            raise ValueError(
                f"Thread ID {thread_id} was not returned by reserve_thread_id()."
            )
        _reserved_thread_ids.remove(thread_id)
        _threads[thread_id] = weakref.ref(
            thread,
            lambda reference, registered_id=thread_id: _remove_thread(
                registered_id, reference
            ),
        )


def _remove_thread(
    thread_id: int,
    reference: weakref.ReferenceType[_MessageTarget],
) -> None:
    """Remove a dead thread without racing a later registry update."""
    with _registry_lock:
        if _threads.get(thread_id) is reference:
            del _threads[thread_id]


__all__ = ["invoke_async", "reserve_thread_id"]
