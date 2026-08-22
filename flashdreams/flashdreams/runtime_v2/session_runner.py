# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-visible-thread lifecycle, input fan-out, and frame compositing."""

import logging
import queue
import threading

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.presentation_cordinator import (
    PresentationCordinator,
    WhenFull,
)
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)


def _contains(events: UserInputEvents, event_type: type[UserInputEventData]) -> bool:
    """Return whether any event in ``events`` carries ``event_type`` data."""
    return any(
        isinstance(event.get_event_data(), event_type) for event in events.get_events()
    )


def run_session(
    session: ISession,
    window: IClientWindow,
    *,
    steps: int | None = None,
    max_pending: int = 2,
    when_full: WhenFull = WhenFull.BLOCK,
) -> None:
    """Run one session and its user-visible-threads against one client window.

    The model-generation-thread and additional user-visible-threads registered
    by :meth:`ISession.init` run independently at their own frequencies. The
    main-program-thread owns the window, fans each input event out to every
    user-visible-thread, and composites their latest enabled frames in the
    explicit order configured by
    :meth:`ISession.set_layer_order_via_thread_id`.

    Args:
        session: Uninitialized session to run.
        window: Client window supplying input and presenting frames.
        steps: Number of model-generation-thread steps; ``None`` runs until
            close input.
        max_pending: Maximum composited frames waiting for UI presentation.
        when_full: Whether full UI presentation waits or drops its oldest frame.

    Raises:
        ValueError: ``steps`` is negative, or ``max_pending`` is not positive.
    """
    if steps is not None and steps < 0:
        raise ValueError(f"steps must be >= 0 or None, got {steps}.")
    if max_pending <= 0:
        raise ValueError(f"max_pending must be > 0, got {max_pending}.")

    thread_manager = session._ensure_thread_manager()
    presentation_cordinator = session.get_presentation_cordinator(
        max_pending=max_pending,
        when_full=when_full,
    )
    try:
        session.init()
        thread_manager._require_model_generation_thread()
        layer_order = thread_manager._get_layer_order()
    except Exception:
        thread_manager._stop()
        session.close()
        raise

    event_buffer = EventBuffer()

    stop = threading.Event()
    model_generation_finished = threading.Event()
    failures: queue.Queue[BaseException] = queue.Queue()
    presentation_failures: queue.Queue[Exception] = queue.Queue()
    tick_seconds = 1.0 / session.session_desc.frames_per_second_for_ui
    dropped_frames = 0

    def read_input() -> None:
        events = window.get_user_input_events()
        event_buffer.append(events)
        presentation_cordinator._set_generation(event_buffer.generation)
        if _contains(events, CloseUserInputEventData):
            stop.set()

    presentation_index = 0
    try:
        window.open(session.session_desc)
        read_input()
        if not stop.is_set():
            thread_manager._start(
                event_buffer=event_buffer,
                stop=stop,
                failure=failures,
                finished=model_generation_finished,
                max_steps=steps,
            )
            while not stop.wait(tick_seconds):
                read_input()
                event_buffer.collect_garbage()
                model_generation_was_finished = model_generation_finished.is_set()

                presentable = presentation_cordinator._take_presentable_results(
                    thread_manager._freeze(),
                    event_buffer.generation,
                    presentation_index,
                    session.session_desc.output_layout,
                    session.main_generation_thread_id,
                    layer_order,
                )
                for result in presentable:
                    dropped_frames += presentation_cordinator._push(
                        result,
                        window.write,
                    )
                    presentation_index += 1
                presentation_cordinator._drain(window.write)

                if model_generation_was_finished:
                    stop.set()
    except Exception as error:
        presentation_failures.put(error)
    finally:
        stop.set()
        try:
            thread_manager._stop()
        except Exception as error:
            failures.put(error)
        event_buffer.clear()

        try:
            window.close()
        except Exception as error:
            presentation_failures.put(error)

        try:
            session.close()
        except Exception as error:
            failures.put(error)

    if dropped_frames:
        _LOGGER.warning(
            "Dropped %d frames the client window could not keep up with.",
            dropped_frames,
        )

    if not failures.empty():
        raise failures.get()
    if not presentation_failures.empty():
        raise presentation_failures.get()


__all__ = ["PresentationCordinator", "WhenFull", "run_session"]
