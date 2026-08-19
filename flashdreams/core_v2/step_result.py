# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step result implementation."""

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.core_v2.time_window import TimeWindow
from flashdreams.core_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True)
class StepResult:
    """Generated output returned by one inference step."""

    step_index: int
    output: Tensor | None = None
    frame_count: int = 0
    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    output_window: TimeWindow = TimeWindow()
    metrics: dict[str, float | int] = field(default_factory=dict)
