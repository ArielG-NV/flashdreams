# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for main input-buffer cursors and garbage collection."""

import pytest
from numpy import uint64

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _key(timestamp: int, key: str) -> UserInputEvent:
    return UserInputEvent(
        timestamp=uint64(timestamp),
        event_data=KeyboardUserInputEventData(
            key=key, state=KeyboardInputState.PRESSED
        ),
    )


def _read(event_buffer: EventBuffer, thread_id: int) -> list[UserInputEvent]:
    events, _ = event_buffer.read(thread_id)
    return events.get_events()


def test_each_thread_reads_its_own_unread_suffix() -> None:
    event_buffer = EventBuffer()
    event_buffer.register(10)
    event_buffer.register(20)
    first = _key(0, "a")
    second = _key(1, "b")
    event_buffer.append(UserInputEvents([first, second]))

    assert _read(event_buffer, 10) == [first, second]
    assert _read(event_buffer, 10) == []
    assert _read(event_buffer, 20) == [first, second]


def test_unseen_thread_starts_at_the_beginning_of_retained_events() -> None:
    event_buffer = EventBuffer()
    event = _key(0, "a")
    event_buffer.append(UserInputEvents([event]))

    assert _read(event_buffer, 10) == [event]

    next_event = _key(1, "b")
    event_buffer.append(UserInputEvents([next_event]))
    assert _read(event_buffer, 10) == [next_event]


def test_garbage_collection_shifts_every_thread_cursor() -> None:
    event_buffer = EventBuffer()
    event_buffer.register(10)
    event_buffer.register(20)
    first = _key(0, "a")
    second = _key(1, "b")
    event_buffer.append(UserInputEvents([first, second]))

    assert _read(event_buffer, 10) == [first, second]
    assert event_buffer.collect_garbage() == 0
    assert _read(event_buffer, 20) == [first, second]
    assert event_buffer.collect_garbage() == 2
    assert event_buffer.retained_count() == 0

    third = _key(2, "c")
    event_buffer.append(UserInputEvents([third]))
    assert _read(event_buffer, 20) == [third]
