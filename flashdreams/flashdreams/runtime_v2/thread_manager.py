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

"""Session thread ownership, communication, lifecycle, and compositing."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any, final

import torch
from torch import Tensor

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.internal_thread import InternalThread
from flashdreams.runtime_v2.step_result import PresentationMode, StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_MODEL_GENERATION_THREAD_ID = 0
"""Reserved identifier for the session's generation worker."""


class _ThreadManager:
    """Own session threads and coordinate communication between them."""

    def __init__(self) -> None:
        self._threads: dict[int, InternalThread[Any]] = {}
        self._registry_frozen = False
        self._generation = 0
        self._generation_lock = threading.Lock()

    @final
    def _get_model_generation_thread_id(self) -> int:
        """Return the reserved main-generation thread identifier."""
        return _MODEL_GENERATION_THREAD_ID

    @final
    def _register_thread(
        self,
        thread: InternalThread[Any],
        thread_id: int,
    ) -> None:
        """Register an auxiliary thread.

        Args:
            thread: Constructed worker owned by this manager.
            thread_id: Positive manager-unique identifier.

        Raises:
            RuntimeError: Registration is frozen or ``thread`` already has a parent.
            TypeError: ``thread_id`` or ``thread`` has an invalid type.
            ValueError: ``thread_id`` is reserved, negative, or already registered.
        """
        if isinstance(thread_id, bool) or not isinstance(thread_id, int):
            raise TypeError("thread_id must be an integer.")
        if thread_id == _MODEL_GENERATION_THREAD_ID:
            raise ValueError("Thread ID 0 is reserved for main generation.")
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
    def _register_main_thread(self, thread: InternalThread[Any]) -> None:
        """Register the runtime's main-generation adapter."""
        self._register(thread, _MODEL_GENERATION_THREAD_ID)

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
        """Start all registered threads."""
        for thread_id, worker in self._freeze().items():
            is_main = thread_id == _MODEL_GENERATION_THREAD_ID
            worker._start(
                thread_id=thread_id,
                event_buffer=event_buffer,
                stop=stop,
                failure=failure,
                finished=finished if is_main else None,
                max_steps=max_steps if is_main else None,
            )

    @final
    def _composite_next(
        self,
        generation: int,
        presentation_index: int,
        output_layout: VideoTensorLayout,
    ) -> StepResult | None:
        """Record eligible frames and composite the visible ones."""
        self._set_generation(generation)
        composed: Tensor | None = None
        recorded: list[tuple[InternalThread[Any], Tensor]] = []
        threads = self._freeze()
        for thread_id in sorted(threads):
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
        if composed is None:
            return None
        return StepResult(
            step_index=presentation_index,
            output=_frame_to_layout(composed, output_layout),
            frame_count=1,
            output_layout=output_layout,
        )

    @final
    def _stop(self) -> None:
        """Stop all threads and discard pending messages."""
        threads = self._freeze().values()
        for worker in threads:
            worker._stop_accepting_messages()
        for worker in threads:
            worker._join()
        for worker in threads:
            worker._empty_message_queue()

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
