# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe fan-out buffer for session input events."""

import threading

from flashdreams.runtime_v2.user_input_event import (
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class EventBuffer:
    """Deliver every retained input event once to every known thread."""

    def __init__(self) -> None:
        self._events: list[UserInputEvent] = []
        self._base_index = 0
        self._event_indexes: dict[int, int] = {}
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def generation(self) -> int:
        """Return the current reset generation."""
        with self._lock:
            return self._generation

    def clear(self) -> None:
        """Remove all retained events and registered readers."""
        with self._lock:
            self._events.clear()
            self._event_indexes.clear()
            self._base_index = 0

    def append(self, events: UserInputEvents) -> None:
        """Append one arrival-ordered batch from the client window."""
        received = events.get_events()
        with self._lock:
            self._events.extend(received)
            self._generation += sum(
                isinstance(event.get_event_data(), ResetUserInputEventData)
                for event in received
            )

    def read(self, thread_id: int) -> tuple[UserInputEvents, int]:
        """Return events not yet read by ``thread_id`` and advance its cursor.

        An unknown thread starts at the beginning of the retained buffer.

        Args:
            thread_id: Session thread, added as a reader on first use.

        Returns:
            Unread events and the reset generation current at the read.

        """
        with self._lock:
            event_index = self._event_indexes.setdefault(thread_id, self._base_index)
            relative_index = event_index - self._base_index
            events = list(self._events[relative_index:])
            self._event_indexes[thread_id] = self._base_index + len(self._events)
            return UserInputEvents(events), self._generation

    def register(self, thread_id: int) -> None:
        """Register a thread to read events."""
        with self._lock:
            self._event_indexes.setdefault(thread_id, self._base_index)

    def unregister(self, thread_id: int) -> None:
        """Stop retaining events on behalf of ``thread_id``."""
        with self._lock:
            self._event_indexes.pop(thread_id, None)

    def collect_garbage(self) -> int:
        """Discard the prefix read by every known thread.

        Returns:
            Number of discarded events.
        """
        with self._lock:
            if not self._event_indexes:
                removed = len(self._events)
            else:
                removed = min(self._event_indexes.values()) - self._base_index
            if removed <= 0:
                return 0
            del self._events[:removed]
            self._base_index += removed
            return removed

    def retained_count(self) -> int:
        """Return the number of events still retained for readers."""
        with self._lock:
            return len(self._events)


__all__ = ["EventBuffer"]
