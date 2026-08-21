# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session worker lifecycle, input fan-out, and frame compositing."""

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable
from enum import Enum

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)
_INTERRUPT_POLL_SECONDS = 0.1


class WhenFull(Enum):
    """What UI presentation does when its bounded frame queue is full."""

    BLOCK = "block"
    """Wait for presentation capacity so every generated frame is retained."""

    DROP_OLDEST = "drop_oldest"
    """Discard the oldest queued frame in favor of the newest frame."""


class _PresentationBuffer:
    """Hold composited single frames until the client window writes them."""

    def __init__(self, max_pending: int, when_full: WhenFull) -> None:
        self._max_pending = max_pending
        self._when_full = when_full
        self._frames: deque[StepResult] = deque()

    def push(
        self,
        frame: StepResult,
        write: Callable[[StepResult], None],
    ) -> int:
        """Queue one composite, applying the configured full-buffer policy.

        Args:
            frame: Single composited frame to present.
            write: Synchronous window write used to apply back-pressure.

        Returns:
            Number of frames dropped to make room.
        """
        if len(self._frames) >= self._max_pending:
            if self._when_full is WhenFull.DROP_OLDEST:
                self._frames.popleft()
                dropped = 1
            else:
                write(self._frames.popleft())
                dropped = 0
        else:
            dropped = 0
        self._frames.append(frame)
        return dropped

    def drain(self, write: Callable[[StepResult], None]) -> None:
        """Write every queued composite in presentation order."""
        while self._frames:
            write(self._frames.popleft())


def _contains(events: UserInputEvents, event_type: type[UserInputEventData]) -> bool:
    """Return whether any event in ``events`` carries ``event_type`` data."""
    return any(
        isinstance(event.get_event_data(), event_type) for event in events.get_events()
    )


def _join_interruptibly(thread: threading.Thread) -> None:
    """Join a thread without indefinitely blocking Python signal handling."""
    while thread.is_alive():
        thread.join(timeout=_INTERRUPT_POLL_SECONDS)


def run_session(
    session: ISession,
    window: IClientWindow,
    *,
    steps: int | None = None,
    max_pending: int = 2,
    when_full: WhenFull = WhenFull.BLOCK,
) -> None:
    """Run one session and its workers against one client window.

    Thread zero delegates to :meth:`ISession.step`. Auxiliary workers registered
    by :meth:`ISession.init` run independently at their own frequencies. The I/O
    thread owns the window, fans each input event out to every worker, and
    composites their latest enabled frames in ascending thread-ID order.

    Args:
        session: Uninitialized session to run.
        window: Client window supplying input and presenting frames.
        steps: Number of main-generation steps; ``None`` runs until close input.
        max_pending: Maximum composited frames waiting for UI presentation.
        when_full: Whether full UI presentation waits or drops its oldest frame.

    Raises:
        ValueError: ``steps`` is negative, or ``max_pending`` is not positive.
    """
    if steps is not None and steps < 0:
        raise ValueError(f"steps must be >= 0 or None, got {steps}.")
    if max_pending <= 0:
        raise ValueError(f"max_pending must be > 0, got {max_pending}.")

    session._register_main_generation_thread()
    try:
        session.init()
    except Exception:
        session._stop_threads()
        session.close()
        raise

    event_buffer = EventBuffer()

    stop = threading.Event()
    opened = threading.Event()
    workers_started = threading.Event()
    main_finished = threading.Event()
    failures: queue.Queue[Exception] = queue.Queue()
    io_failures: queue.Queue[Exception] = queue.Queue()
    tick_seconds = 1.0 / session.session_desc.frames_per_second_for_ui
    presentation_buffer = _PresentationBuffer(max_pending, when_full)
    dropped_frames = 0

    def read_input() -> None:
        events = window.get_user_input_events()
        event_buffer.append(events)
        if _contains(events, CloseUserInputEventData):
            stop.set()

    def run_io() -> None:
        nonlocal dropped_frames
        presentation_index = 0
        try:
            window.open(session.session_desc)
            read_input()
            opened.set()

            # Before we start trying to read input and composite frames we should wait for the workers to be started.
            workers_started.wait()
            while not stop.wait(tick_seconds):
                read_input()
                event_buffer.collect_garbage()
                main_was_finished = main_finished.is_set()

                composite = session._composite_next(
                    event_buffer.generation,
                    presentation_index,
                )
                if composite is not None:
                    dropped_frames += presentation_buffer.push(
                        composite,
                        window.write,
                    )
                    presentation_index += 1
                presentation_buffer.drain(window.write)

                if main_was_finished:
                    stop.set()
        except Exception as error:
            io_failures.put(error)
            stop.set()
        finally:
            opened.set()
            workers_started.set()
            try:
                window.close()
            except Exception as error:
                io_failures.put(error)
            stop.set()

    io_thread = threading.Thread(target=run_io, name="flashdreams-io")
    io_thread.start()
    try:
        # Before we start threads working on the window we should wait for the window to be opened.
        opened.wait()
        if io_failures.empty() and not stop.is_set():
            session._start_threads(
                event_buffer=event_buffer,
                stop=stop,
                failure=failures,
                finished=main_finished,
                max_steps=steps,
            )
    except Exception as error:
        failures.put(error)
        stop.set()
    finally:
        workers_started.set()

    try:
        _join_interruptibly(io_thread)
    finally:
        stop.set()
        io_thread.join()
        session._stop_threads()
        event_buffer.clear()

        try:
            session.close()
        except Exception as error:
            failures.put(error)

    if dropped_frames:
        _LOGGER.warning(
            "Dropped %d frames the presentation thread could not keep up with.",
            dropped_frames,
        )

    if not failures.empty():
        raise failures.get()
    if not io_failures.empty():
        raise io_failures.get()


__all__ = ["WhenFull", "run_session"]
