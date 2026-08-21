# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-owned session worker behavior."""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, final

from torch import Tensor

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import CloseUserInputEventData
from flashdreams.runtime_v2.user_input_events import UserInputEvents

if TYPE_CHECKING:
    from flashdreams.runtime_v2.thread_manager import _ThreadManager

StateT = TypeVar("StateT")


@dataclass(slots=True)
class Message(Generic[StateT]):
    """One state operation waiting to run on its owning thread."""

    operation: Callable[[StateT], None]
    """Callable applied to the thread's state."""


@dataclass(frozen=True, slots=True)
class _LatestStep:
    """Published step plus the session generation that produced it."""

    generation: int
    result: StepResult


@dataclass(frozen=True, slots=True)
class _PresentedFrame:
    """Presented frame plus the session generation that selected it."""

    generation: int
    frame: Tensor


class InternalThread(ABC, Generic[StateT]):
    """Provide runtime-owned behavior for the public worker interface."""

    def __init__(self, *, state: StateT, frequency: int) -> None:
        """Initialize a session worker without starting it.

        Args:
            state: Mutable state owned by this worker.
            frequency: Maximum steps per second. Zero runs without pacing.

        Raises:
            TypeError: ``frequency`` is not an integer.
            ValueError: ``frequency`` is negative.
        """
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise TypeError("frequency must be an integer.")
        if frequency < 0:
            raise ValueError("frequency must be >= 0.")
        self.state = state
        self.frequency = frequency
        self.user_events = UserInputEvents([])
        self.latest_step: StepResult | None = None
        self._message_queue: queue.Queue[Message[StateT]] = queue.Queue()
        self._latest: _LatestStep | None = None
        self._pending_steps: deque[_LatestStep] = deque()
        self._last_presented_frame: _PresentedFrame | None = None
        self._latest_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._accepting_messages = True
        self._native_thread: threading.Thread | None = None
        self._thread_manager: _ThreadManager | None = None

    @final
    def _get_thread_manager(self) -> _ThreadManager:
        """Return the runtime manager that owns this thread.

        Raises:
            RuntimeError: The thread has not been registered.
        """
        if self._thread_manager is None:
            raise RuntimeError("Thread has not been registered with a manager.")
        return self._thread_manager

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Produce one result from the events received since the previous step."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state before the first step of a new generation."""
        ...

    @final
    def _enqueue_message(self, operation: Callable[[StateT], None]) -> None:
        """Queue an operation to run before the next ``step`` or ``step_ui``.

        Args:
            operation: Callable that receives the worker-owned state.

        Raises:
            RuntimeError: The worker is shutting down.
        """
        with self._lifecycle_lock:
            if not self._accepting_messages:
                raise RuntimeError("Thread is shutting down.")
            self._message_queue.put(Message(operation=operation))

    @final
    def _start(
        self,
        *,
        thread_id: int,
        event_buffer: EventBuffer,
        stop: threading.Event,
        failure: queue.Queue[BaseException],
        finished: threading.Event | None = None,
        max_steps: int | None = None,
    ) -> None:
        with self._lifecycle_lock:
            if self._native_thread is not None:
                raise RuntimeError("Thread has already been started.")
            self._native_thread = threading.Thread(
                target=self._run,
                kwargs={
                    "thread_id": thread_id,
                    "event_buffer": event_buffer,
                    "stop": stop,
                    "failure": failure,
                    "finished": finished,
                    "max_steps": max_steps,
                },
                name=f"flashdreams-session-{thread_id}",
            )
            self._native_thread.start()

    @final
    def _run(
        self,
        *,
        thread_id: int,
        event_buffer: EventBuffer,
        stop: threading.Event,
        failure: queue.Queue[BaseException],
        finished: threading.Event | None,
        max_steps: int | None,
    ) -> None:
        step_index = 0
        steps_run = 0
        generation = 0
        last_step_started: float | None = None
        event_buffer.register(thread_id)
        try:
            while not stop.is_set() and (max_steps is None or steps_run < max_steps):
                self._run_message_batch()
                events, read_generation = event_buffer.read(thread_id)
                self.user_events = events
                if _contains_close(events):
                    stop.set()
                    break
                if read_generation != generation:
                    self.reset()
                    self._clear_last_presented_frame()
                    step_index = 0
                    generation = read_generation
                if self._is_finished():
                    break
                last_step_started = self._pace(last_step_started, stop)
                if stop.is_set():
                    break
                result = self.step(step_index, events)
                if stop.is_set():
                    break
                with self._latest_lock:
                    self.latest_step = result
                    self._latest = _LatestStep(generation, result)
                    self._pending_steps.append(self._latest)
                step_index += 1
                steps_run += 1
        except BaseException as error:
            failure.put(error)
            stop.set()
        finally:
            try:
                self._close()
            except BaseException as error:
                failure.put(error)
                stop.set()
            with self._lifecycle_lock:
                self._accepting_messages = False
            self._empty_message_queue()
            event_buffer.unregister(thread_id)
            if finished is not None:
                finished.set()

    @final
    def _run_message_batch(self) -> None:
        for _ in range(self._message_queue.qsize()):
            try:
                message = self._message_queue.get_nowait()
            except queue.Empty:
                return
            result = message.operation(self.state)
            if result is not None:
                raise TypeError("Message operations must return None.")

    @final
    def _pace(self, last_step_started: float | None, stop: threading.Event) -> float:
        if self.frequency == 0 or last_step_started is None:
            return time.monotonic()
        earliest_start = last_step_started + 1.0 / self.frequency
        stop.wait(max(0.0, earliest_start - time.monotonic()))
        return time.monotonic()

    @final
    def _snapshot_latest(self) -> _LatestStep | None:
        """Return the latest completed step and its generation."""
        with self._latest_lock:
            return self._latest

    @final
    def _take_pending_steps(self) -> list[_LatestStep]:
        """Take every completed step not yet consumed by presentation."""
        with self._latest_lock:
            pending = list(self._pending_steps)
            self._pending_steps.clear()
            return pending

    @final
    def _bind_thread_manager(self, thread_manager: _ThreadManager) -> None:
        """Bind this thread to its parent manager."""
        if self._thread_manager is not None:
            raise RuntimeError("Thread is already registered with a manager.")
        self._thread_manager = thread_manager

    @final
    def _set_last_presented_frame(self, generation: int, frame: Tensor) -> None:
        """Record the frame most recently selected by the compositor."""
        with self._latest_lock:
            self._last_presented_frame = _PresentedFrame(generation, frame)

    @final
    def _snapshot_last_presented_frame(self) -> _PresentedFrame | None:
        """Return the frame most recently selected by the compositor."""
        with self._latest_lock:
            return self._last_presented_frame

    @final
    def _clear_last_presented_frame(self) -> None:
        """Discard the frame retained from the previous generation."""
        with self._latest_lock:
            self._last_presented_frame = None
            self._pending_steps.clear()

    @final
    def _join(self, timeout: float | None = None) -> bool:
        """Wait for this worker to stop.

        Args:
            timeout: Maximum seconds to wait; ``None`` waits indefinitely.

        Returns:
            Whether the worker stopped before the timeout.
        """
        native_thread = self._native_thread
        if native_thread is None:
            return True
        native_thread.join(timeout=timeout)
        return not native_thread.is_alive()

    @final
    def _stop_accepting_messages(self) -> None:
        with self._lifecycle_lock:
            self._accepting_messages = False

    def _close(self) -> None:
        """Release worker-owned resources on the native worker thread."""
        return

    def _is_finished(self) -> bool:
        """Return whether this worker has completed its finite workload."""
        return False

    @final
    def _empty_message_queue(self) -> None:
        while True:
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                return


def _contains_close(events: UserInputEvents) -> bool:
    return any(
        isinstance(event.get_event_data(), CloseUserInputEventData)
        for event in events.get_events()
    )


__all__ = ["InternalThread", "Message"]
