# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for presentation-time video post-processing."""

from __future__ import annotations

import threading

import pytest
import torch

from flashdreams.demo.post_processing import (
    HWC_UINT8_FORMAT,
    PostProcessingChunk,
    PostProcessingFormat,
    PostProcessingFrame,
    PostProcessingOutput,
    PostProcessingPipeline,
    PostProcessingPipelineStep,
)
from flashdreams.demo.post_processing_utils import (
    HWC_UINT8_POST_PROCESSING_STEP,
    INVERT_POST_PROCESSING_STEP,
)
from flashdreams.runtime import StepResult

pytestmark = pytest.mark.ci_cpu


def _collect_frames(
    pipeline: PostProcessingPipeline,
    video: torch.Tensor,
) -> list[PostProcessingFrame]:
    stream = pipeline.process_frames(
        video,
        format=PostProcessingFormat(
            layout="tchw",
            value_range="minus_one_one",
        ),
        chunk_index=7,
    )
    try:
        return list(stream)
    finally:
        stream.close()


def test_steps_propagate_format_through_hwc_uint8_conversion() -> None:
    def add_red(
        partition: PostProcessingChunk | PostProcessingFrame,
    ) -> PostProcessingOutput:
        assert isinstance(partition, PostProcessingFrame)
        output = partition.data.clone()
        output[0] = 1.0
        return PostProcessingOutput(data=output, format=partition.format)

    def add_blue(
        partition: PostProcessingChunk | PostProcessingFrame,
    ) -> PostProcessingOutput:
        assert isinstance(partition, PostProcessingFrame)
        output = partition.data.clone()
        output[2] = 1.0
        return PostProcessingOutput(data=output, format=partition.format)

    pipeline = PostProcessingPipeline(
        (
            PostProcessingPipelineStep(
                input_kind="frame",
                operation=add_red,
            ),
            PostProcessingPipelineStep(
                input_kind="frame",
                operation=add_blue,
            ),
            HWC_UINT8_POST_PROCESSING_STEP,
        )
    )
    video = torch.full((2, 3, 4, 5), -1.0)

    frames = _collect_frames(pipeline, video)

    assert [frame.format for frame in frames] == [
        HWC_UINT8_FORMAT,
        HWC_UINT8_FORMAT,
    ]
    assert all(frame.data.shape == (4, 5, 3) for frame in frames)
    assert all(frame.data.dtype == torch.uint8 for frame in frames)
    assert all(frame.data[0, 0].tolist() == [255, 0, 255] for frame in frames)


def test_step_rejects_tensor_output_without_format() -> None:
    def tensor_only(
        partition: PostProcessingChunk | PostProcessingFrame,
    ) -> PostProcessingOutput:
        return partition.data  # ty: ignore[invalid-return-type]

    pipeline = PostProcessingPipeline(
        (
            PostProcessingPipelineStep(
                input_kind="frame",
                operation=tensor_only,
            ),
        )
    )

    with pytest.raises(TypeError, match="must return PostProcessingOutput"):
        _collect_frames(pipeline, torch.zeros((1, 3, 2, 2)))


def test_chunk_step_can_change_layout_and_propagate_frame_format() -> None:
    def to_thwc(
        partition: PostProcessingChunk | PostProcessingFrame,
    ) -> PostProcessingOutput:
        assert isinstance(partition, PostProcessingChunk)
        return PostProcessingOutput(
            data=partition.data.permute(0, 2, 3, 1),
            format=PostProcessingFormat(
                layout="thwc",
                value_range="minus_one_one",
            ),
        )

    pipeline = PostProcessingPipeline(
        (
            PostProcessingPipelineStep(
                input_kind="chunk",
                operation=to_thwc,
            ),
        )
    )
    video = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

    frames = _collect_frames(pipeline, video)

    assert [frame.format.layout for frame in frames] == ["hwc", "hwc"]
    assert torch.equal(
        frames[1].data,
        video[1].permute(1, 2, 0),
    )


def test_next_frame_processes_while_first_frame_is_presentable() -> None:
    second_started = threading.Event()
    release_second = threading.Event()

    def staged_operation(
        partition: PostProcessingChunk | PostProcessingFrame,
    ) -> PostProcessingOutput:
        assert isinstance(partition, PostProcessingFrame)
        if partition.frame_number == 1:
            second_started.set()
            if not release_second.wait(timeout=5.0):
                raise TimeoutError("test did not release the second frame")
        return PostProcessingOutput(
            data=partition.data + 1,
            format=partition.format,
        )

    pipeline = PostProcessingPipeline(
        (
            PostProcessingPipelineStep(
                input_kind="frame",
                operation=staged_operation,
            ),
        )
    )
    stream = pipeline.process_frames(
        torch.zeros((2, 3, 2, 2)),
        format=PostProcessingFormat(
            layout="tchw",
            value_range="minus_one_one",
        ),
        chunk_index=3,
        max_in_flight_frames=2,
    )
    try:
        first = next(stream)
        assert first.frame_number == 0
        assert second_started.wait(timeout=1.0)
        assert not release_second.is_set()
        release_second.set()
        second = next(stream)
        assert second.frame_number == 1
    finally:
        release_second.set()
        stream.close()


def test_pipeline_rejects_chunk_barrier_after_frame_steps() -> None:
    def identity(
        partition: PostProcessingChunk | PostProcessingFrame,
    ) -> PostProcessingOutput:
        return PostProcessingOutput(data=partition.data, format=partition.format)

    with pytest.raises(ValueError, match="barriers"):
        PostProcessingPipeline(
            (
                PostProcessingPipelineStep(
                    input_kind="frame",
                    operation=identity,
                ),
                PostProcessingPipelineStep(
                    input_kind="chunk",
                    operation=identity,
                ),
            )
        )


def test_step_result_full_screen_effect_preserves_raw_chunk() -> None:
    video = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    original = StepResult.from_video_chunk(
        step_index=4,
        video_chunk=video,
        layout="tchw",
    )
    registered = original.register_post_processing_pipeline(
        PostProcessingPipeline((INVERT_POST_PROCESSING_STEP,)),
        chunk_index=9,
    )

    stream = registered.post_processed_frame_partitions(ensure_hwc_uint8=True)
    try:
        frames = list(stream)
    finally:
        stream.close()

    assert original.post_processing_pipeline is None
    assert registered.video_chunk is video
    assert [(frame.chunk_index, frame.frame_number) for frame in frames] == [
        (9, 0),
        (9, 1),
    ]
    assert all(frame.format == HWC_UINT8_FORMAT for frame in frames)
    assert torch.equal(
        registered.video_hwc_uint8(),
        (-video).add(1).mul(127.5).round().to(torch.uint8).permute(0, 2, 3, 1),
    )
