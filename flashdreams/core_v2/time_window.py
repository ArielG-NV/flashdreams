# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-window implementation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeWindow:
    """Half-open time window in seconds since session start."""

    start_s: float = 0
    end_s: float = 0

    def __init__(self, start_s: float = 0, end_s: float = 0):
        if not math.isfinite(start_s) or not math.isfinite(end_s):
            raise ValueError("TimeWindow bounds must be finite seconds.")
        if start_s < 0 or end_s < 0:
            raise ValueError("TimeWindow bounds must be non-negative.")
        if end_s < start_s:
            raise ValueError("TimeWindow.end_s must be >= start_s.")
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "end_s", end_s)

    def contains(self, timestamp_s: float) -> bool:
        """Return whether ``timestamp_s`` falls within this half-open window."""
        return self.start_s <= timestamp_s < self.end_s
