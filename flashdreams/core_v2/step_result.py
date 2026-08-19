# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step result implementation."""

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.core_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True)
class StepResult:
    """Generated output returned by one inference step."""

    # number of the step
    step_index: int
    # output tensor
    output: Tensor
    # chunk size in frames
    frame_count: int
    # output tensor layout
    output_layout: VideoTensorLayout
    # metrics
    metrics: dict[str, float | int]
