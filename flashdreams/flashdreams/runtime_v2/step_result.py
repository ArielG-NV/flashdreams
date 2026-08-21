# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output of one generation step."""

from dataclasses import dataclass, field
from enum import Enum

from torch import Tensor

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class PresentationMode(Enum):
    """Control how a step result participates in frame presentation."""

    showPresentation = "showPresentation"
    """Update the last-presented frame and composite it into the client backbuffer."""

    hidePresentation = "hidePresentation"
    """Update the last-presented frame without affecting the client backbuffer."""

    disablePresentation = "disablePresentation"
    """Skip frame extraction, updates to the last-frame presented, and backbuffer compositing.
    Primarily used for testing without blitting the frame to the client backbuffer."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """Generated output returned by one inference step."""

    step_index: int
    """Zero-based index of the step that produced this result."""
    output: Tensor
    """Generated frames, laid out as ``output_layout`` says."""
    frame_count: int
    """Number of frames in ``output``."""
    output_layout: VideoTensorLayout
    """Layout of ``output``."""
    metrics: dict[str, float | int] = field(default_factory=dict)
    """Measurements for this step, such as timings, keyed by name."""
    presentation_mode: PresentationMode = PresentationMode.showPresentation
    """How the latest frame updates shared state and the client backbuffer."""


__all__ = ["PresentationMode", "StepResult"]
