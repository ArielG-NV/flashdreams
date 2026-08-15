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
    PostProcessingFrame,
    PostProcessingInput,
    PostProcessingOutput,
    PostProcessingPipelineStep,
)


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


__all__ = [
    "FrameTensorLayout",
    "HWC_UINT8_POST_PROCESSING_STEP",
    "INVERT_POST_PROCESSING_STEP",
    "PixelValueRange",
    "frame_tensor_to_hwc_uint8",
]
