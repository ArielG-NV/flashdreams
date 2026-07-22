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

"""Scripted MIRA controls shared by CLI demos and tests."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

import nvtx

from flashdreams.serving.webrtc.runtime import WebRTCStepResult
from mira_integration.configs.schema import MiraModelMetadata
from mira_integration.webrtc.session import MiraInferenceRuntime

MIRA_KEYS = frozenset(
    {"W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"}
)
DEFAULT_ACTION_SCRIPT = "W@1,W+D@1,W@1,W+A@1"
"""A short deterministic lap-like demonstration (400 ms of scripted controls)."""

_T = TypeVar("_T")


@nvtx.annotate("mira.scripted.parse_action_script")
def parse_action_script(
    value: str,
    *,
    valid_keys: frozenset[str] = MIRA_KEYS,
    fps: int = 60,
    frames_per_chunk: int = 1,
) -> list[list[str]]:
    """Expand ``KEY+KEY@100MS`` segments into per-chunk held controls."""
    if not value.strip():
        raise ValueError("action_script must contain at least one segment")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be > 0")
    timeline: list[list[str]] = []
    for raw_segment in value.split(","):
        segment = raw_segment.strip()
        try:
            key_spec, duration_spec = segment.rsplit("@", 1)
            duration_100ms = int(duration_spec)
        except ValueError as exc:
            raise ValueError(
                f"invalid action segment {segment!r}; expected KEY+KEY@100MS"
            ) from exc
        if duration_100ms <= 0:
            raise ValueError(f"action duration must be positive in {segment!r}")
        keys = [key.strip() for key in key_spec.split("+") if key.strip()]
        unknown = sorted(set(keys) - valid_keys)
        if unknown:
            raise ValueError(f"unknown MIRA key(s) in {segment!r}: {unknown}")
        count = math.ceil(duration_100ms * fps / (10 * frames_per_chunk))
        timeline.extend([keys] * count)
    return timeline


@nvtx.annotate("mira.scripted.player_one_browser_controls")
def player_one_browser_controls(
    held_checkpoint_keys: Iterable[str],
    *,
    metadata: MiraModelMetadata,
) -> tuple[frozenset[str] | None, ...]:
    """Apply scripted checkpoint keys to player one as browser-key input."""
    checkpoint_to_browser = {
        binding.checkpoint_key: binding.browser_key
        for binding in metadata.input_key_map
    }
    browser_keys = frozenset(
        checkpoint_to_browser[key] for key in held_checkpoint_keys
    )
    return (browser_keys,) + (None,) * (metadata.player_count - 1)


@nvtx.annotate("mira.scripted.run_action_script")
async def run_action_script(
    runtime: MiraInferenceRuntime,
    script: str,
    *,
    metadata: MiraModelMetadata,
    fps: int,
    on_chunk: Callable[[WebRTCStepResult], Awaitable[None]],
) -> None:
    """Publish scripted controls and notify ``on_chunk`` after each render."""
    controls = parse_action_script(
        script,
        valid_keys=frozenset(
            binding.checkpoint_key for binding in metadata.input_key_map
        ),
        fps=fps,
        frames_per_chunk=metadata.frames_per_chunk,
    )
    for held in controls:
        runtime.publish_player_keys(
            player_one_browser_controls(held, metadata=metadata)
        )
        await on_chunk(await runtime.render_next_chunk())


__all__ = [
    "DEFAULT_ACTION_SCRIPT",
    "MIRA_KEYS",
    "parse_action_script",
    "player_one_browser_controls",
    "run_action_script",
]
