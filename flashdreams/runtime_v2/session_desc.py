# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session information API."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionDesc:
    """Sink-facing metadata container."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Declared tensor layout for generated video results."""

    frames_per_second_for_ui: int = 60
    """frame rate to display ui-thread frames at."""

    frames_per_second_for_step: int = 30
    """frame rate to display model-generation thread frames at."""

    video_width: int = 1280
    """Output video width in pixels."""

    video_height: int = 720
    """Output video height in pixels."""

    metadata: Mapping[str, object] = field(default_factory=dict)
    """Additional flags to share downstream."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.frames_per_second_for_ui) or self.frames_per_second_for_ui <= 0:
            raise ValueError("SessionDesc.frames_per_second_for_ui must be > 0 when set.")
        if not math.isfinite(self.frames_per_second_for_step) or self.frames_per_second_for_step <= 0:
            raise ValueError("SessionDesc.frames_per_second_for_step must be > 0 when set.")
        if self.video_width <= 0:
            raise ValueError("SessionDesc.video_width must be > 0 when set.")
        if self.video_height <= 0:
            raise ValueError("SessionDesc.video_height must be > 0 when set.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
