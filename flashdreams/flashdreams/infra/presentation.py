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

"""Format-aware, frame-streaming presentation post-processing."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

from flashdreams.infra.postprocess import VideoTensorLayout

FrameTensorLayout: TypeAlias = Literal["chw", "hwc"]
"""Supported layouts for one RGB frame."""

PixelValueRange: TypeAlias = Literal["minus_one_one", "zero_one", "uint8"]
"""Declared numeric range of RGB tensor values."""

ChunkTensorLayout: TypeAlias = VideoTensorLayout | Literal["thwc"]
"""Supported layouts for a generated video chunk."""

PostProcessingTensorLayout: TypeAlias = ChunkTensorLayout | FrameTensorLayout
"""Tensor layouts carried between presentation steps."""

PostProcessingInputKind: TypeAlias = Literal["chunk", "frame"]
"""Granularity consumed by one presentation step."""


@dataclass(frozen=True, slots=True)
class PostProcessingFormat:
    """Layout and numeric-range contract carried beside tensor data."""

    layout: PostProcessingTensorLayout
    """Current tensor layout."""

    value_range: PixelValueRange
    """Current numeric interpretation of RGB pixels."""

    def __post_init__(self) -> None:
        if self.layout not in (
            "tchw",
            "thwc",
            "btchw",
            "bcthw",
            "bvtchw",
            "chw",
            "hwc",
        ):
            raise ValueError(f"unsupported post-processing layout: {self.layout!r}")
        if self.value_range not in ("minus_one_one", "zero_one", "uint8"):
            raise ValueError(f"unsupported pixel value range: {self.value_range!r}")


@dataclass(frozen=True, slots=True)
class PostProcessingChunk:
    """Whole generated chunk passed to a chunk-scoped step."""

    data: Tensor
    """Video tensor in format."""

    format: PostProcessingFormat
    """Current tensor format."""

    chunk_index: int
    """Application-provided index of the generated chunk."""

    @property
    def frame_count(self) -> int:
        """Return the number of frames in the chunk."""
        return int(self.data.shape[_video_time_dim(self.format.layout)])


@dataclass(frozen=True, slots=True)
class PostProcessingFrame:
    """One frame partition passed through frame-scoped steps."""

    data: Tensor
    """Single-frame tensor in format."""

    format: PostProcessingFormat
    """Current tensor format after the most recent step."""

    chunk_index: int
    """Application-provided index of the generated chunk."""

    frame_number: int
    """Zero-based frame number within the chunk."""

    ready_event: Any | None = None
    """CUDA event recorded after this frame's final operation."""


PostProcessingInput: TypeAlias = PostProcessingChunk | PostProcessingFrame
"""Input partition accepted by a presentation step."""


@dataclass(frozen=True, slots=True)
class PostProcessingOutput:
    """Tensor data and its format produced by one pipeline operation."""

    data: Tensor
    """Replacement tensor data."""

    format: PostProcessingFormat
    """Layout and numeric interpretation of ``data``."""


PostProcessingOperation: TypeAlias = Callable[
    [PostProcessingInput], PostProcessingOutput
]
"""Data-and-format operation used by one presentation step."""


@dataclass(frozen=True, kw_only=True, slots=True)
class PostProcessingPipelineStep:
    """One explicitly chunk- or frame-scoped presentation operation."""

    input_kind: PostProcessingInputKind
    """Partition granularity passed to operation."""

    operation: PostProcessingOperation
    """Operation that returns replacement data and its matching format."""

    name: str | None = None
    """Optional diagnostic name for validation errors."""

    def __post_init__(self) -> None:
        if self.input_kind not in ("chunk", "frame"):
            raise ValueError(
                "PostProcessingPipelineStep.input_kind must be 'chunk' or 'frame'."
            )
        if not callable(self.operation):
            raise TypeError("PostProcessingPipelineStep.operation must be callable.")

    def process(self, partition: PostProcessingInput) -> PostProcessingInput:
        """Apply the operation and propagate its declared output format."""
        expected_type = (
            PostProcessingChunk if self.input_kind == "chunk" else PostProcessingFrame
        )
        if not isinstance(partition, expected_type):
            raise TypeError(
                f"{self._display_name()} expects a {self.input_kind} partition, "
                f"got {type(partition).__name__}."
            )
        output = self.operation(partition)
        if not isinstance(output, PostProcessingOutput):
            raise TypeError(
                f"{self._display_name()} must return PostProcessingOutput, "
                f"got {type(output).__name__}."
            )
        if not isinstance(output.data, Tensor):
            raise TypeError(
                f"{self._display_name()} output data must be a torch.Tensor, "
                f"got {type(output.data).__name__}."
            )
        if not isinstance(output.format, PostProcessingFormat):
            raise TypeError(
                f"{self._display_name()} output format must be PostProcessingFormat."
            )
        _validate_partition_format(
            data=output.data,
            format=output.format,
            input_kind=self.input_kind,
            step_name=self._display_name(),
        )
        if isinstance(partition, PostProcessingChunk):
            return replace(
                partition,
                data=output.data,
                format=output.format,
            )
        return replace(
            partition,
            data=output.data,
            format=output.format,
            ready_event=None,
        )

    def _display_name(self) -> str:
        return self.name or type(self).__name__


class PostProcessingFrameStream(Iterator[PostProcessingFrame]):
    """Bounded ordered frame stream with one background processing worker."""

    def __init__(
        self,
        *,
        frames: Iterator[PostProcessingFrame],
        steps: tuple[PostProcessingPipelineStep, ...],
        max_in_flight_frames: int,
        source_event: Any | None,
    ) -> None:
        if max_in_flight_frames <= 0:
            raise ValueError("max_in_flight_frames must be > 0.")
        self._frames = frames
        self._steps = steps
        self._source_event = source_event
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="flashdreams-post-processing",
        )
        self._pending: deque[Future[PostProcessingFrame]] = deque()
        self._closed = False
        for _ in range(max_in_flight_frames):
            if not self._submit_next():
                break

    def __iter__(self) -> PostProcessingFrameStream:
        return self

    def __next__(self) -> PostProcessingFrame:
        if not self._pending:
            self.close()
            raise StopIteration
        future = self._pending.popleft()
        try:
            frame = future.result()
        except BaseException:
            self.close()
            raise
        self._submit_next()
        if not self._pending:
            self._executor.shutdown(wait=False)
        return frame

    def close(self) -> None:
        """Cancel queued frame work and release the worker."""
        if self._closed:
            return
        self._closed = True
        while self._pending:
            self._pending.popleft().cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _submit_next(self) -> bool:
        if self._closed:
            return False
        try:
            frame = next(self._frames)
        except StopIteration:
            return False
        self._pending.append(self._executor.submit(self._process_frame, frame))
        return True

    def _process_frame(self, frame: PostProcessingFrame) -> PostProcessingFrame:
        if self._source_event is not None:
            torch.cuda.current_stream(frame.data.device).wait_event(self._source_event)
        for step in self._steps:
            processed = step.process(frame)
            if not isinstance(processed, PostProcessingFrame):
                raise TypeError("Frame-scoped steps must return frame partitions.")
            frame = processed
        if frame.data.is_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(frame.data.device))
            frame = replace(frame, ready_event=ready_event)
        return frame


class PostProcessingPipeline:
    """Ordered presentation operations with explicit chunk barriers."""

    def __init__(
        self,
        steps: Iterable[PostProcessingPipelineStep] = (),
    ) -> None:
        self.steps = tuple(steps)
        seen_frame_step = False
        for step in self.steps:
            if not isinstance(step, PostProcessingPipelineStep):
                raise TypeError(
                    "PostProcessingPipeline steps must be "
                    "PostProcessingPipelineStep instances."
                )
            if step.input_kind == "frame":
                seen_frame_step = True
            elif seen_frame_step:
                raise ValueError(
                    "Chunk-scoped steps are barriers and must precede frame-scoped "
                    "steps."
                )

    steps: tuple[PostProcessingPipelineStep, ...]
    """Immutable ordered post-processing steps."""

    def append(self, step: PostProcessingPipelineStep) -> PostProcessingPipeline:
        """Return a pipeline with step appended."""
        return PostProcessingPipeline((*self.steps, step))

    def ensure_hwc_uint8(self) -> PostProcessingPipeline:
        """Append the bundled HWC uint8 step when the output needs conversion."""
        if self.steps and self.steps[-1] is HWC_UINT8_POST_PROCESSING_STEP:
            return self
        return self.append(HWC_UINT8_POST_PROCESSING_STEP)

    def process_frames(
        self,
        video_chunk: Tensor,
        *,
        format: PostProcessingFormat,
        chunk_index: int,
        batch_index: int = 0,
        view_index: int = 0,
        max_in_flight_frames: int = 2,
    ) -> PostProcessingFrameStream:
        """Start processing one chunk as a bounded ordered frame stream.

        Chunk-scoped steps finish before the stream starts because they are
        barriers. Frame-scoped work is prefetched on one worker so the next
        frame can process while the current frame is presented.

        Args:
            video_chunk: Generated video tensor in format.
            format: Initial layout and pixel range.
            chunk_index: Application-provided generated chunk index.
            batch_index: Batch element selected for frame-scoped work.
            view_index: View selected for frame-scoped work.
            max_in_flight_frames: Maximum submitted frames including the next
                frame being processed.

        Returns:
            Ordered closeable frame iterator.
        """
        if chunk_index < 0:
            raise ValueError("chunk_index must be >= 0.")
        _validate_partition_format(
            data=video_chunk,
            format=format,
            input_kind="chunk",
            step_name="pipeline input",
        )
        chunk = PostProcessingChunk(
            data=video_chunk,
            format=format,
            chunk_index=chunk_index,
        )
        for step in self._chunk_steps:
            processed = step.process(chunk)
            if not isinstance(processed, PostProcessingChunk):
                raise TypeError("Chunk-scoped steps must return chunk partitions.")
            chunk = processed
        source_event = None
        if chunk.data.is_cuda:
            source_event = torch.cuda.Event()
            source_event.record(torch.cuda.current_stream(chunk.data.device))
        frames = _partition_chunk_frames(
            chunk,
            batch_index=batch_index,
            view_index=view_index,
        )
        return PostProcessingFrameStream(
            frames=frames,
            steps=self._frame_steps,
            max_in_flight_frames=max_in_flight_frames,
            source_event=source_event,
        )

    @property
    def _chunk_steps(self) -> tuple[PostProcessingPipelineStep, ...]:
        return tuple(step for step in self.steps if step.input_kind == "chunk")

    @property
    def _frame_steps(self) -> tuple[PostProcessingPipelineStep, ...]:
        return tuple(step for step in self.steps if step.input_kind == "frame")


HWC_UINT8_FORMAT = PostProcessingFormat(layout="hwc", value_range="uint8")
"""Canonical frame format consumed by presentation backends."""


def _frame_to_hwc_uint8(partition: PostProcessingInput) -> PostProcessingOutput:
    from flashdreams.infra.presentation_utils import frame_tensor_to_hwc_uint8

    if not isinstance(partition, PostProcessingFrame):
        raise TypeError("HWC uint8 conversion requires a frame partition.")
    if partition.format.layout not in ("chw", "hwc"):
        raise ValueError(
            "HWC uint8 conversion requires a CHW or HWC frame, "
            f"got {partition.format.layout!r}."
        )
    return PostProcessingOutput(
        data=frame_tensor_to_hwc_uint8(
            partition.data,
            layout=partition.format.layout,
            value_range=partition.format.value_range,
        ),
        format=HWC_UINT8_FORMAT,
    )


HWC_UINT8_POST_PROCESSING_STEP = PostProcessingPipelineStep(
    input_kind="frame",
    operation=_frame_to_hwc_uint8,
    name="frame-to-hwc-uint8",
)
"""Reusable frame conversion step for user-authored pipelines."""


def infer_post_processing_format(
    video_chunk: Tensor,
    *,
    layout: VideoTensorLayout,
) -> PostProcessingFormat:
    """Infer the default model-output format from tensor dtype and layout."""
    value_range: PixelValueRange = (
        "uint8" if video_chunk.dtype == torch.uint8 else "minus_one_one"
    )
    return PostProcessingFormat(layout=layout, value_range=value_range)


def _partition_chunk_frames(
    chunk: PostProcessingChunk,
    *,
    batch_index: int,
    view_index: int,
) -> Iterator[PostProcessingFrame]:
    frame_format = _frame_format_from_chunk(chunk.format)
    for frame_number in range(chunk.frame_count):
        data = _select_frame(
            chunk.data,
            layout=chunk.format.layout,
            frame_number=frame_number,
            batch_index=batch_index,
            view_index=view_index,
        )
        yield PostProcessingFrame(
            data=data,
            format=frame_format,
            chunk_index=chunk.chunk_index,
            frame_number=frame_number,
        )


def _frame_format_from_chunk(
    format: PostProcessingFormat,
) -> PostProcessingFormat:
    if format.layout in ("tchw", "btchw", "bcthw", "bvtchw"):
        return replace(format, layout="chw")
    if format.layout == "thwc":
        return replace(format, layout="hwc")
    raise ValueError(f"unsupported chunk layout: {format.layout!r}")


def _select_frame(
    data: Tensor,
    *,
    layout: PostProcessingTensorLayout,
    frame_number: int,
    batch_index: int,
    view_index: int,
) -> Tensor:
    if layout in ("tchw", "thwc"):
        return data[frame_number]
    if layout == "btchw":
        return data[batch_index, frame_number]
    if layout == "bcthw":
        return data[batch_index, :, frame_number]
    if layout == "bvtchw":
        return data[batch_index, view_index, frame_number]
    raise ValueError(f"unsupported chunk layout: {layout!r}")


def _video_time_dim(layout: PostProcessingTensorLayout) -> int:
    if layout in ("tchw", "thwc"):
        return 0
    if layout == "btchw":
        return 1
    if layout in ("bcthw", "bvtchw"):
        return 2
    raise ValueError(f"unsupported chunk layout: {layout!r}")


def _validate_partition_format(
    *,
    data: Tensor,
    format: PostProcessingFormat,
    input_kind: PostProcessingInputKind,
    step_name: str,
) -> None:
    if format.value_range == "uint8":
        if data.dtype != torch.uint8:
            raise ValueError(
                f"{step_name} declared uint8 values for dtype {data.dtype}."
            )
    elif not data.is_floating_point():
        raise ValueError(
            f"{step_name} declared {format.value_range!r} values for "
            f"non-floating dtype {data.dtype}."
        )

    if input_kind == "frame":
        valid = (format.layout == "chw" and data.ndim == 3 and data.shape[0] == 3) or (
            format.layout == "hwc" and data.ndim == 3 and data.shape[-1] == 3
        )
        if not valid:
            raise ValueError(
                f"{step_name} declared frame format {format.layout!r} for "
                f"shape {tuple(data.shape)}."
            )
        return
    expected_shape = {
        "tchw": (4, 1),
        "thwc": (4, -1),
        "btchw": (5, 2),
        "bcthw": (5, 1),
        "bvtchw": (6, 3),
    }.get(format.layout)
    valid = False
    if expected_shape is not None:
        expected_ndim, channel_dim = expected_shape
        valid = data.ndim == expected_ndim and data.shape[channel_dim] == 3
    if not valid:
        raise ValueError(
            f"{step_name} declared chunk format {format.layout!r} for "
            f"shape {tuple(data.shape)}."
        )


__all__ = [
    "ChunkTensorLayout",
    "HWC_UINT8_FORMAT",
    "HWC_UINT8_POST_PROCESSING_STEP",
    "PostProcessingChunk",
    "PostProcessingFormat",
    "PostProcessingFrame",
    "PostProcessingFrameStream",
    "PostProcessingOutput",
    "PostProcessingInput",
    "PostProcessingInputKind",
    "PostProcessingPipeline",
    "PostProcessingPipelineStep",
    "PostProcessingTensorLayout",
    "infer_post_processing_format",
]
