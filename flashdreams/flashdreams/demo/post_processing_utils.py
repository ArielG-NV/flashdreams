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

"""Device-preserving frame conversion and reusable presentation steps."""

from __future__ import annotations

import torch
from torch import Tensor

from flashdreams.demo.post_processing import (
    HWC_UINT8_FORMAT,
    FrameTensorLayout,
    PixelValueRange,
    PostProcessingChunk,
    PostProcessingFormat,
    PostProcessingFrame,
    PostProcessingInput,
    PostProcessingOutput,
    PostProcessingPipelineStep,
)
from flashdreams.infra.postprocess.base import (
    VideoChunk,
    VideoPostProcessorConfig,
    VideoSpec,
    VideoTensorLayout,
    concatenate_video_chunks,
    infer_video_spec_from_tensor_shape,
)

_FLASH_VSR_PRESET = "flashvsr-v1.1-sparse-2.0"
"""Stable FlashVSR preset used by the demo presentation pipeline."""

_DLSS_PRESET = "rtx-super-resolution"
"""RTX Video Super Resolution preset exposed by the demo as DLSS upscaling."""


def frame_tensor_to_hwc_uint8(
    frame: Tensor,
    *,
    layout: FrameTensorLayout,
    value_range: PixelValueRange,
) -> Tensor:
    """Convert one RGB tensor to contiguous HWC uint8 on its device.

    Args:
        frame: RGB frame in layout.
        layout: Channel placement in frame.
        value_range: Numeric interpretation of the input pixels.

    Returns:
        Contiguous [H, W, 3] uint8 tensor on the input device.
    """
    if layout == "chw":
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(
                f"layout='chw' requires shape [3, H, W], got {tuple(frame.shape)}."
            )
        output = frame.permute(1, 2, 0)
    elif layout == "hwc":
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"layout='hwc' requires shape [H, W, 3], got {tuple(frame.shape)}."
            )
        output = frame
    else:
        raise ValueError(f"unsupported frame layout: {layout!r}")

    if value_range == "uint8":
        if output.dtype != torch.uint8:
            raise ValueError("value_range='uint8' requires a torch.uint8 tensor.")
        return output.detach().contiguous()
    if not output.is_floating_point():
        raise ValueError(
            f"value_range={value_range!r} requires a floating-point tensor."
        )
    if value_range == "minus_one_one":
        output = (output.clamp(-1.0, 1.0) + 1.0) * 127.5
    elif value_range == "zero_one":
        output = output.clamp(0.0, 1.0) * 255.0
    else:
        raise ValueError(f"unsupported pixel value range: {value_range!r}")
    return output.round().to(torch.uint8).detach().contiguous()


def _frame_to_hwc_uint8(partition: PostProcessingInput) -> PostProcessingOutput:
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


def _invert_frame(partition: PostProcessingInput) -> PostProcessingOutput:
    data = partition.data
    if partition.format.value_range == "uint8":
        data = 255 - data
    elif partition.format.value_range == "zero_one":
        data = 1.0 - data
    else:
        data = -data
    return PostProcessingOutput(data=data, format=partition.format)


INVERT_POST_PROCESSING_STEP = PostProcessingPipelineStep(
    input_kind="frame",
    operation=_invert_frame,
    name="full-screen-invert",
)
"""Reusable full-frame color-inversion step."""


def _resolve_postprocessor_config(preset: str) -> VideoPostProcessorConfig:
    from flashdreams.plugins.registry import resolve_postprocess_preset

    return resolve_postprocess_preset(preset)


def _postprocess_video_chunk(
    partition: PostProcessingInput,
    *,
    preset: str,
) -> PostProcessingOutput:
    if not isinstance(partition, PostProcessingChunk):
        raise TypeError("Video upscaling requires a chunk partition.")
    if partition.format.value_range != "minus_one_one":
        raise ValueError(
            "Video upscaling requires float RGB values in the [-1, 1] range."
        )
    if partition.format.layout not in ("tchw", "btchw", "bcthw", "bvtchw"):
        raise ValueError(
            "Video upscaling requires a supported video layout, "
            f"got {partition.format.layout!r}."
        )

    layout: VideoTensorLayout = partition.format.layout
    config = _resolve_postprocessor_config(preset)
    spec = infer_video_spec_from_tensor_shape(
        partition.data,
        layout=layout,
    )
    session = config.setup().start(spec)
    session.prepare()
    output_chunks = session.process(
        VideoChunk(
            tensor=partition.data,
            layout=layout,
            metadata={"chunk_index": partition.chunk_index},
        )
    )
    output_chunks.extend(session.flush())
    if not output_chunks:
        raise RuntimeError(f"Post-processing preset {preset!r} emitted no frames.")

    output_layout = output_chunks[0].layout
    return PostProcessingOutput(
        data=concatenate_video_chunks(output_chunks, layout=output_layout),
        format=PostProcessingFormat(
            layout=output_layout,
            value_range="minus_one_one",
        ),
    )


def _flash_vsr_upscale(partition: PostProcessingInput) -> PostProcessingOutput:
    return _postprocess_video_chunk(partition, preset=_FLASH_VSR_PRESET)


def _flash_vsr_output_spec(input_spec: VideoSpec) -> VideoSpec:
    return _resolve_postprocessor_config(_FLASH_VSR_PRESET).output_spec(input_spec)


FLASH_VSR_POST_PROCESSING_STEP = PostProcessingPipelineStep(
    input_kind="chunk",
    operation=_flash_vsr_upscale,
    output_spec_operation=_flash_vsr_output_spec,
    name="flashvsr-upscale",
)
"""Per-chunk 2x FlashVSR presentation upscaling step."""


def _dlss_upscale(partition: PostProcessingInput) -> PostProcessingOutput:
    return _postprocess_video_chunk(partition, preset=_DLSS_PRESET)


def _dlss_output_spec(input_spec: VideoSpec) -> VideoSpec:
    return _resolve_postprocessor_config(_DLSS_PRESET).output_spec(input_spec)


DLSS_POST_PROCESSING_STEP = PostProcessingPipelineStep(
    input_kind="chunk",
    operation=_dlss_upscale,
    output_spec_operation=_dlss_output_spec,
    name="dlss-upscale",
)
"""Per-chunk 2x DLSS presentation step backed by RTX Video Super Resolution."""


__all__ = [
    "DLSS_POST_PROCESSING_STEP",
    "FLASH_VSR_POST_PROCESSING_STEP",
    "FrameTensorLayout",
    "HWC_UINT8_POST_PROCESSING_STEP",
    "INVERT_POST_PROCESSING_STEP",
    "PixelValueRange",
    "frame_tensor_to_hwc_uint8",
]
