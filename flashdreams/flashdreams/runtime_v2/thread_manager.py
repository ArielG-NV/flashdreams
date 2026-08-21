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

import torch
from torch import Tensor

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.internal_thread import InternalThread
from flashdreams.runtime_v2.step_result import PresentationMode, StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_MODEL_GENERATION_THREAD_ID = 0
"""Reserved identifier for the session's model-generation-thread."""

_THREAD_STOP_TIMEOUT_SECONDS = 30.0
"""Maximum total wait for all user-visible-threads to stop."""


class _ThreadManager:
    """Own a session's user-visible-threads and coordinate communication."""

    def __init__(self) -> None:
        self._threads: dict[int, InternalThread[Any]] = {}
        self._registry_frozen = False
        self._generation = 0
        self._generation_lock = threading.Lock()

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
    def _get_last_presented_frame(self, thread_id: int) -> Tensor | None:
        """Return a thread's most recently presented frame.

        The returned ``[C, H, W]`` tensor is shared with the producing thread and
        must be treated as read-only. ``None`` means no enabled frame from that
        thread has been presented in the current generation.

        Args:
            thread_id: Identifier of the thread whose frame is requested.

        Returns:
            Latest presented frame, or ``None`` before its first presentation.

        Raises:
            KeyError: No thread is registered under ``thread_id``.
        """
        presented = self._get_thread(thread_id)._snapshot_last_presented_frame()
        if presented is None or presented.generation != self._snapshot_generation():
            return None
        return presented.frame

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
    def _take_presentable_results(
        self,
        generation: int,
        presentation_index: int,
        output_layout: VideoTensorLayout,
    ) -> list[StepResult]:
        """Take original model results or one newly composited UI frame.

        A model-only run is also the lossless file/benchmark path, so every
        original result must retain its full frame batch and metrics. Sessions
        with additional user-visible-threads are presentation streams: they
        consume each thread's newest result and composite one current frame.
        """
        self._set_generation(generation)
        threads = self._freeze()

        # Preserve every model result when there are no additional
        # user-visible-threads to composite.
        if set(threads) == {_MODEL_GENERATION_THREAD_ID}:
            model_generation_thread = threads[_MODEL_GENERATION_THREAD_ID]
            results: list[StepResult] = []
            for latest in model_generation_thread._take_pending_steps():
                if latest.generation != generation:
                    continue
                result = latest.result
                if result.presentation_mode is PresentationMode.disablePresentation:
                    continue
                frame = _latest_frame(result)
                if frame is not None:
                    model_generation_thread._set_last_presented_frame(generation, frame)
                if result.presentation_mode is PresentationMode.showPresentation:
                    results.append(result)
            return results

        composed: Tensor | None = None
        recorded: list[tuple[InternalThread[Any], Tensor]] = []
        has_new_visible_result = False
        for thread_id in sorted(threads):
            pending = threads[thread_id]._take_pending_steps()
            has_new_visible_result = has_new_visible_result or any(
                latest.generation == generation
                and latest.result.presentation_mode is PresentationMode.showPresentation
                for latest in pending
            )
            latest = threads[thread_id]._snapshot_latest()
            if latest is None or latest.generation != generation:
                continue
            mode = latest.result.presentation_mode
            if mode is PresentationMode.disablePresentation:
                continue
            frame = _latest_frame(latest.result)
            recorded.append((threads[thread_id], frame))
            if mode is PresentationMode.showPresentation:
                composed = _composite_frame(composed, frame)
        for thread, frame in recorded:
            thread._set_last_presented_frame(generation, frame)
        if composed is None or not has_new_visible_result:
            return []
        return [
            StepResult(
                step_index=presentation_index,
                output=_frame_to_layout(composed, output_layout),
                frame_count=1,
                output_layout=output_layout,
                metrics=_model_metrics(threads, generation),
            )
        ]

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

    @final
    def _set_generation(self, generation: int) -> None:
        """Publish the current input generation."""
        with self._generation_lock:
            self._generation = generation

    def _snapshot_generation(self) -> int:
        with self._generation_lock:
            return self._generation


def _latest_frame(result: StepResult) -> Tensor:
    """Return the newest frame as ``[C, H, W]``."""
    output = result.output
    if result.output_layout is VideoTensorLayout.tchw:
        frame = output[-1]
    elif result.output_layout is VideoTensorLayout.btchw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("btchw compositing requires a batch size of one.")
        frame = output[0, -1]
    elif result.output_layout is VideoTensorLayout.bcthw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("bcthw compositing requires a batch size of one.")
        frame = output[0, :, -1]
    elif result.output_layout is VideoTensorLayout.bvtchw:
        if output.ndim != 6 or output.shape[:2] != (1, 1):
            raise ValueError(
                "bvtchw compositing requires one batch and one video view."
            )
        frame = output[0, 0, -1]
    else:
        raise ValueError(f"Unsupported compositing layout: {result.output_layout}.")
    if frame.ndim != 3 or frame.shape[0] not in (1, 3, 4):
        raise ValueError("A composited frame must have one, three, or four channels.")
    return frame


def _model_metrics(
    threads: dict[int, InternalThread[Any]], generation: int
) -> dict[str, float | int]:
    """Return current model metrics for a composited presentation frame."""
    model = threads.get(_MODEL_GENERATION_THREAD_ID)
    latest = model._snapshot_latest() if model is not None else None
    if latest is None or latest.generation != generation:
        return {}
    return dict(latest.result.metrics)


def _composite_frame(bottom: Tensor | None, top: Tensor) -> Tensor:
    """Place one RGB or RGBA frame over the accumulated RGB frame."""
    color = top[:3]
    if color.shape[0] == 1:
        color = color.repeat(3, 1, 1)
    if bottom is not None and color.shape[1:] != bottom.shape[1:]:
        raise ValueError("All composited frames must have the same dimensions.")
    if bottom is not None and (
        color.device != bottom.device or color.dtype != bottom.dtype
    ):
        raise ValueError("All composited frames must have the same device and dtype.")
    if top.shape[0] != 4:
        return color
    if not top.is_floating_point():
        raise ValueError("RGBA compositing requires a floating-point tensor.")

    if bottom is None:
        fill_value = -1.0 if color.is_floating_point() else 0
        bottom = torch.full_like(color, fill_value)
    alpha = top[3:4].to(device=bottom.device, dtype=torch.float32)
    alpha = alpha.clamp(0.0, 1.0).to(bottom.dtype)
    return color * alpha + bottom * (1.0 - alpha)


def _frame_to_layout(frame: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Add singleton time, batch, and view dimensions for ``layout``."""
    if layout is VideoTensorLayout.tchw:
        return frame.unsqueeze(0)
    if layout is VideoTensorLayout.btchw:
        return frame.unsqueeze(0).unsqueeze(0)
    if layout is VideoTensorLayout.bcthw:
        return frame.unsqueeze(0).unsqueeze(2)
    if layout is VideoTensorLayout.bvtchw:
        return frame.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported compositing layout: {layout}.")


__all__: list[str] = []
