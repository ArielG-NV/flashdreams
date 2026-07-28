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

"""MIRA frame-request and media-push timing records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class MiraFramePushTiming:
    """Track one generated chunk from frame request through media push."""

    first_frame_number: int
    """Zero-based number of the first requested frame."""

    requested_at_s: float
    """Monotonic time when generation of the first frame was requested."""

    completed_frame_number: int | None = None
    """Exclusive cumulative frame number completed by this media push."""

    media_push_finished_at_s: float | None = None
    """Monotonic time when the media push returned."""

    chunk_index: int | None = None
    """Autoregressive chunk index associated with the pushed frames."""

    real_time_budget_ms: float | None = None
    """Playback-time budget for the pushed frames."""

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a JSON-serializable timing record."""
        return asdict(self)


__all__ = ["MiraFramePushTiming"]
