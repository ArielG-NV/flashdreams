# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-owned session behavior."""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, final

import torch
from torch import Tensor

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_MAIN_GENERATION_THREAD_ID = 0
"""Reserved identifier for the session's generation worker."""


class InternalSession(ABC):
    """Provide runtime-owned behavior for the public session interface."""

    _threads: dict[int, IThread[Any]]
    """Workers registered for this session, initialized on first use."""

    _thread_registry_frozen: bool
    """Whether the runtime has stopped accepting worker registration."""

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to create this session."""
        ...

    @abstractmethod
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Produce one result for ``step_index``."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset per-generation state."""
        ...

    @abstractmethod
    def is_finished(self) -> bool:
        """Report whether main generation should stop before its next step."""
        ...

    @final
    def _register_main_generation_thread(self) -> None:
        """Register the adapter that runs :meth:`step` as thread zero."""
        self._ensure_thread_registry()
        if _MAIN_GENERATION_THREAD_ID in self._threads:
            raise ValueError("Main generation thread is already registered.")
        self._threads[_MAIN_GENERATION_THREAD_ID] = _MainGenerationThread(
            state=self,
            frequency=self.session_desc.frames_per_second_for_step,
        )

    @final
    def _freeze_thread_registry(self) -> dict[int, IThread[Any]]:
        """Freeze registration and return workers in compositing order."""
        self._ensure_thread_registry()
        self._thread_registry_frozen = True
        return dict(self._threads)

    @final
    def _start_threads(
        self,
        *,
        event_buffer: EventBuffer,
        stop: threading.Event,
        failure: queue.Queue[BaseException],
        finished: threading.Event,
        max_steps: int | None,
    ) -> None:
        """Start every registered worker.

        ``finished`` and ``max_steps`` apply only to the main-generation worker.

        Args:
            event_buffer: Shared input-event buffer.
            stop: Shared shutdown signal.
            failure: Queue receiving worker failures.
            finished: Signal set when main generation finishes.
            max_steps: Main-generation step limit; ``None`` runs until shutdown.
        """
        for thread_id, worker in self._freeze_thread_registry().items():
            is_main_generation = thread_id == _MAIN_GENERATION_THREAD_ID
            worker._start(
                thread_id=thread_id,
                event_buffer=event_buffer,
                stop=stop,
                failure=failure,
                finished=finished if is_main_generation else None,
                max_steps=max_steps if is_main_generation else None,
            )

    @final
    def _composite_next(
        self,
        generation: int,
        presentation_index: int,
    ) -> StepResult | None:
        """Composite the latest frame from each current worker.

        Args:
            generation: Session generation whose frames may be composited.
            presentation_index: Step index assigned to the composite.

        Returns:
            Composite in the session output layout, or ``None`` when no worker
            has an enabled frame for ``generation``.
        """
        composed: Tensor | None = None
        threads = self._freeze_thread_registry()
        for thread_id in sorted(threads):
            latest = threads[thread_id]._snapshot_latest()
            if (
                latest is None
                or latest.generation != generation
                or latest.result.disabled
            ):
                continue
            composed = _composite_frame(composed, _latest_frame(latest.result))
        if composed is None:
            return None
        output_layout = self.session_desc.output_layout
        return StepResult(
            step_index=presentation_index,
            output=_frame_to_layout(composed, output_layout),
            frame_count=1,
            output_layout=output_layout,
        )

    @final
    def _stop_threads(self) -> None:
        """Stop every registered worker and discard its pending messages."""
        threads = self._freeze_thread_registry().values()
        for worker in threads:
            worker._stop_accepting_messages()
        for worker in threads:
            worker._join()
        for worker in threads:
            worker._empty_message_queue()

    @final
    def _ensure_thread_registry(self) -> None:
        """Initialize the registry on its first use."""
        if not hasattr(self, "_threads"):
            self._threads = {}
            self._thread_registry_frozen = False


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


class _MainGenerationThread(IThread[InternalSession]):
    """Adapt the session's generation methods to the worker contract."""

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Delegate one generation step to the session."""
        return self.state.step(step_index, events)

    def reset(self) -> None:
        """Delegate a generation reset to the session."""
        self.state.reset()

    def _is_finished(self) -> bool:
        """Let a finite session end without a client close event."""
        return self.state.is_finished()


__all__ = ["InternalSession"]
