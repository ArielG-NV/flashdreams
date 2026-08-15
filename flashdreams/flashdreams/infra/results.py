# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated inference result contracts shared by runtimes and consumers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.presentation import (
    PostProcessingFrameStream,
    PostProcessingPipeline,
    infer_post_processing_format,
)
from flashdreams.infra.time import TimeWindow

if TYPE_CHECKING:
    from flashdreams.infra.video_output import LazyRGBFrame


@dataclass(frozen=True, kw_only=True, slots=True)
class StepResult:
    """Generated output and metadata returned by one inference step.

    Video results use :meth:`from_video_chunk`, which records a required tensor
    layout and derives the frame count once. Non-video results may use the
    regular constructor without a layout.
    """

    __hash__ = None

    step_index: int
    output: Any = None
    frame_count: int = 0
    layout: VideoTensorLayout | None = None
    output_window: TimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    post_processing_pipeline: PostProcessingPipeline | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    """Presentation pipeline registered for this generated chunk."""

    post_processing_chunk_index: int | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    """Chunk coordinate supplied to the registered presentation pipeline."""

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepResult.step_index must be >= 0.")
        if self.frame_count < 0:
            raise ValueError("StepResult.frame_count must be >= 0.")
        if (self.post_processing_pipeline is None) != (
            self.post_processing_chunk_index is None
        ):
            raise ValueError(
                "StepResult post-processing pipeline and chunk index must be set "
                "together."
            )
        if (
            self.post_processing_chunk_index is not None
            and self.post_processing_chunk_index < 0
        ):
            raise ValueError("StepResult post-processing chunk index must be >= 0.")
        if self.layout is not None:
            from flashdreams.infra.video_output import infer_video_num_frames

            video_chunk = self.video_chunk
            derived_frame_count = infer_video_num_frames(
                video_chunk,
                layout=self.layout,
            )
            if self.frame_count not in (0, derived_frame_count):
                raise ValueError(
                    "StepResult.frame_count does not match the declared video "
                    f"layout: expected {derived_frame_count}, got {self.frame_count}."
                )
            object.__setattr__(self, "frame_count", derived_frame_count)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @classmethod
    def from_video_chunk(
        cls,
        *,
        step_index: int,
        video_chunk: Tensor,
        layout: VideoTensorLayout,
        output_window: TimeWindow | None = None,
        metadata: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float | int] | None = None,
    ) -> StepResult:
        """Build one layout-aware generated-video result."""
        return cls(
            step_index=step_index,
            output=video_chunk,
            layout=layout,
            output_window=output_window,
            metadata=dict(metadata or {}),
            metrics=dict(metrics or {}),
        )

    @property
    def video_chunk(self) -> Tensor:
        """Return the video tensor or fail if this is not a video result."""
        if self.layout is None:
            raise ValueError("StepResult.layout is required for video output.")
        if not isinstance(self.output, Tensor):
            raise TypeError(
                "A video StepResult requires a torch.Tensor output, "
                f"got {type(self.output).__name__}."
            )
        return self.output

    def iter_lazy_rgb_frames(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
        record_cuda_event: bool = True,
        max_in_flight_frames: int = 2,
    ) -> Iterator[LazyRGBFrame]:
        """Yield lazy RGB handles while frame processing continues in parallel."""
        stream = self.post_processed_frame_partitions(
            batch_index=batch_index,
            view_index=view_index,
            ensure_hwc_uint8=True,
            max_in_flight_frames=max_in_flight_frames,
        )
        return _iter_lazy_rgb_frame_stream(
            stream,
            record_cuda_event=record_cuda_event,
        )

    def lazy_rgb_frames(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
        record_cuda_event: bool = True,
    ) -> list[LazyRGBFrame]:
        """Return all processed frames as lazy RGB handles."""
        return list(
            self.iter_lazy_rgb_frames(
                batch_index=batch_index,
                view_index=view_index,
                record_cuda_event=record_cuda_event,
            )
        )

    def video_hwc_uint8(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
    ) -> Tensor:
        """Return this video result as uint8 HWC frames on its device."""
        from flashdreams.infra.video_output import video_tensor_to_hwc_uint8

        stream = self.post_processed_frame_partitions(
            batch_index=batch_index,
            view_index=view_index,
            ensure_hwc_uint8=True,
        )
        try:
            frames = []
            for frame in stream:
                if frame.ready_event is not None:
                    torch.cuda.current_stream(frame.data.device).wait_event(
                        frame.ready_event
                    )
                frames.append(frame.data)
        finally:
            stream.close()
        if frames:
            return torch.stack(frames)
        return video_tensor_to_hwc_uint8(
            self.video_chunk,
            layout=self._video_layout(),
            batch_index=batch_index,
            view_index=view_index,
        )

    def register_post_processing_pipeline(
        self,
        pipeline: PostProcessingPipeline,
        chunk_index: int,
    ) -> StepResult:
        """Return a result with a presentation pipeline registered for its chunk.

        Args:
            pipeline: Presentation operations to run before encoding or display.
            chunk_index: Chunk coordinate exposed to every pipeline partition.

        Returns:
            Immutable copy carrying the pipeline registration.

        Raises:
            ValueError: This is not a video result or chunk_index is negative.
            TypeError: pipeline is not a PostProcessingPipeline.
        """
        if self.layout is None:
            raise ValueError(
                "A post-processing pipeline can only be registered on video output."
            )
        if not isinstance(pipeline, PostProcessingPipeline):
            raise TypeError("pipeline must be a PostProcessingPipeline.")
        if chunk_index < 0:
            raise ValueError("chunk_index must be >= 0.")
        return replace(
            self,
            post_processing_pipeline=pipeline,
            post_processing_chunk_index=chunk_index,
        )

    def post_processed_frame_partitions(
        self,
        *,
        batch_index: int = 0,
        view_index: int = 0,
        ensure_hwc_uint8: bool = False,
        max_in_flight_frames: int = 2,
    ) -> PostProcessingFrameStream:
        """Start a format-aware ordered stream of processed frame partitions."""
        input_format = infer_post_processing_format(
            self.video_chunk,
            layout=self._video_layout(),
        )
        pipeline = self.post_processing_pipeline or PostProcessingPipeline()
        if ensure_hwc_uint8:
            pipeline = pipeline.ensure_hwc_uint8()
        chunk_index = self.post_processing_chunk_index
        if chunk_index is None:
            chunk_index = self.step_index
        return pipeline.process_frames(
            self.video_chunk,
            format=input_format,
            chunk_index=chunk_index,
            batch_index=batch_index,
            view_index=view_index,
            max_in_flight_frames=max_in_flight_frames,
        )

    def _video_layout(self) -> VideoTensorLayout:
        if self.layout is None:
            raise ValueError("StepResult.layout is required for video output.")
        return self.layout


def _iter_lazy_rgb_frame_stream(
    stream: PostProcessingFrameStream,
    *,
    record_cuda_event: bool,
) -> Iterator[LazyRGBFrame]:
    from flashdreams.infra.video_output import LazyRGBFrame

    try:
        for frame in stream:
            source_event = frame.ready_event if record_cuda_event else None
            yield LazyRGBFrame(
                frame.data.unsqueeze(0),
                0,
                source_event=source_event,
            )
    finally:
        stream.close()


__all__ = ["StepResult"]
