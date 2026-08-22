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

"""Session-owned frame compositing and presentation state."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

from flashdreams.runtime_v2.internal_thread import InternalThread
from flashdreams.runtime_v2.step_result import PresentationMode, StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class WhenFull(Enum):
    """What UI presentation does when its bounded frame queue is full."""

    BLOCK = "block"
    """Wait for presentation capacity so every generated frame is retained."""

    DROP_OLDEST = "drop_oldest"
    """Discard the oldest queued frame in favor of the newest frame."""


@dataclass(slots=True)
class _ContainerState:
    """Mutable state shared by a coordinator and one read-only handle."""

    value: Tensor | None = None
    """Most recently presented frame, or ``None`` when none is available."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    """Lock serializing coordinator writes with handle reads."""


class PresentedFrame:
    """Read-only handle to one user-visible-thread's last presented frame."""

    __slots__ = ("__state",)

    def __init__(self, state: _ContainerState) -> None:
        self.__state = state

    def get(self) -> Tensor | None:
        """Return the underlying last-presented-frame object.

        The returned ``[C, H, W]`` tensor is shared with the producing thread
        and must be treated as read-only. ``None`` means no enabled frame from
        that thread has been presented in the current generation.
        """
        with self.__state.lock:
            return self.__state.value


class PresentationCordinator:
    """Coordinate presentation for the lifetime of one session."""

    def get_last_presented_frame(self, thread_id: int) -> PresentedFrame:
        """Return the read-only last-presented-frame handle for ``thread_id``.

        Args:
            thread_id: Identifier of the registered user-visible-thread.

        Returns:
            Stable handle whose :meth:`PresentedFrame.get` method
            reads the current underlying frame.

        Raises:
            KeyError: No user-visible-thread is registered under ``thread_id``.
        """
        try:
            return self._containers[thread_id][0]
        except KeyError as error:
            raise KeyError(f"No thread is registered with ID {thread_id}.") from error

    def __init__(
        self,
        max_pending: int = 2,
        when_full: WhenFull = WhenFull.BLOCK,
    ) -> None:
        """Initialize presentation queues and per-thread frame containers.

        Args:
            max_pending: Maximum composites waiting for client presentation.
            when_full: Policy applied when the presentation queue is full.

        Raises:
            ValueError: ``max_pending`` is not positive.
        """
        self._frames: deque[StepResult] = deque()
        self._containers: dict[int, tuple[PresentedFrame, _ContainerState]] = {}
        self._generation = 0
        self._configure(max_pending, when_full)

    def _push(
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

    def _drain(self, write: Callable[[StepResult], None]) -> None:
        """Write every queued composite in presentation order."""
        while self._frames:
            write(self._frames.popleft())

    def _configure(self, max_pending: int, when_full: WhenFull) -> None:
        if max_pending <= 0:
            raise ValueError(f"max_pending must be > 0, got {max_pending}.")
        self._max_pending = max_pending
        self._when_full = when_full

    def _register_thread(self, thread_id: int) -> None:
        if thread_id in self._containers:
            raise ValueError(f"Thread ID {thread_id} is already registered.")
        state = _ContainerState()
        self._containers[thread_id] = (PresentedFrame(state), state)

    def _set_generation(self, generation: int) -> None:
        if generation == self._generation:
            return
        self._generation = generation
        for _, state in self._containers.values():
            with state.lock:
                state.value = None

    def _record_last_presented_frame(self, thread_id: int, frame: Tensor) -> None:
        _, state = self._containers[thread_id]
        with state.lock:
            state.value = frame

    def _take_presentable_results(
        self,
        threads: dict[int, InternalThread[object]],
        generation: int,
        presentation_index: int,
        output_layout: VideoTensorLayout,
        main_generation_thread_id: int,
        layer_order: tuple[int, ...],
    ) -> list[StepResult]:
        """Take original model results or one newly composited UI frame."""
        self._set_generation(generation)

        # Preserve every model result when there are no additional
        # user-visible-threads to composite.
        if len(threads) == 1:
            thread_id = layer_order[0]
            model_generation_thread = threads[thread_id]
            results: list[StepResult] = []
            for latest in model_generation_thread._take_pending_steps():
                if latest.generation != generation:
                    continue
                result = latest.result
                if result.presentation_mode is PresentationMode.DISABLE_PRESENTATION:
                    continue
                frame = _latest_frame(result)
                self._record_last_presented_frame(thread_id, frame)
                if result.presentation_mode is PresentationMode.SHOW_PRESENTATION:
                    results.append(result)
            return results

        composed: Tensor | None = None
        recorded: list[tuple[int, Tensor]] = []
        has_new_visible_result = False
        for thread_id in layer_order:
            pending = threads[thread_id]._take_pending_steps()
            has_new_visible_result = has_new_visible_result or any(
                latest.generation == generation
                and latest.result.presentation_mode
                is PresentationMode.SHOW_PRESENTATION
                for latest in pending
            )
            latest = threads[thread_id]._snapshot_latest()
            if latest is None or latest.generation != generation:
                continue
            mode = latest.result.presentation_mode
            if mode is PresentationMode.DISABLE_PRESENTATION:
                continue
            frame = _latest_frame(latest.result)
            recorded.append((thread_id, frame))
            if mode is PresentationMode.SHOW_PRESENTATION:
                composed = _composite_frame(composed, frame)
        for thread_id, frame in recorded:
            self._record_last_presented_frame(thread_id, frame)
        if composed is None or not has_new_visible_result:
            return []
        return [
            StepResult(
                step_index=presentation_index,
                output=_frame_to_layout(composed, output_layout),
                frame_count=1,
                output_layout=output_layout,
                metrics=_model_metrics(threads, main_generation_thread_id, generation),
            )
        ]


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
    threads: dict[int, InternalThread[object]],
    main_generation_thread_id: int,
    generation: int,
) -> dict[str, float | int]:
    """Return current model metrics for a composited presentation frame."""
    model = threads.get(main_generation_thread_id)
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


__all__ = [
    "PresentedFrame",
    "PresentationCordinator",
    "WhenFull",
]
