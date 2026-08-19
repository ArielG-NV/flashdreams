# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step result implementation."""

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True)
class StepResult:
    """Generated output returned by one inference step."""

    step_index: int
    """Number of the step."""
    output: Tensor
    """Output tensor."""
    frame_count: int
    """Chunk size in frames."""
    output_layout: VideoTensorLayout
    """Output tensor layout."""
    metrics: dict[str, float | int]
    """Metrics."""
