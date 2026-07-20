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

"""CLI demo runner for MIRA Mini."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from flashdreams.core.io.disk import preflight_runtime_write_paths
from flashdreams.infra.runner import Runner, RunnerConfig
from mira_integration.pipeline import MiraPipeline, MiraPipelineConfig

MIRA_KEYS = frozenset(
    {"W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"}
)
DEFAULT_ACTION_SCRIPT = "W@6,W+D@4,W@6,W+A@4"
"""A short deterministic lap-like demonstration (20 latent steps)."""


@dataclass(kw_only=True)
class MiraDemoRunnerConfig(RunnerConfig):
    """User-facing configuration for the MIRA Mini example rollout."""

    _target: type["MiraDemoRunner"] = field(default_factory=lambda: MiraDemoRunner)
    pipeline: MiraPipelineConfig
    action_script: str = DEFAULT_ACTION_SCRIPT
    """Comma-separated ``KEY+KEY@STEPS`` segments."""
    n_diffusion_steps: int = 2
    """Sampler steps per generated latent frame (the published default is 2)."""
    seed: int = 0
    """Torch RNG seed used for the autoregressive noise stream."""
    fps: int = 20
    """Output video frame rate."""


class MiraDemoRunner(Runner[MiraDemoRunnerConfig, MiraPipeline]):
    """Generate a fixed-action MIRA Mini rollout and persist MP4 + timings."""

    config: MiraDemoRunnerConfig
    pipeline: MiraPipeline

    def __init__(self, config: MiraDemoRunnerConfig) -> None:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("mira-mini-1b-demo supports one GPU only")
        preflight_runtime_write_paths(output_dir=config.output_dir)
        self.config = config
        self.local_rank = self.global_rank = 0
        self.world_size = 1
        self.is_rank_zero = True
        self.pipeline = config.pipeline.setup().to(device=config.device).eval()

    def run(self) -> None:
        """Run the scripted demo and write an MP4 plus per-step timing JSON."""
        controls = parse_action_script(self.config.action_script)
        torch.manual_seed(self.config.seed)
        cache = self.pipeline.initialize_cache(
            n_diffusion_steps=self.config.n_diffusion_steps
        )
        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float | int]] = []
        try:
            for ar_idx, held in enumerate(controls):
                chunks.append(self.pipeline.generate(ar_idx, cache, input=held).cpu())
                stats = self.pipeline.finalize(ar_idx, cache)
                if stats is not None:
                    stats_history.append({"autoregressive_index": ar_idx, **stats})
        finally:
            self.pipeline.close()

        video = torch.cat(chunks, dim=0).permute(0, 2, 3, 1)
        array = (video.float().clamp(0, 1).numpy() * 255).round().astype("uint8")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = self.config.output_dir / f"{self.config.runner_name}.mp4"
        try:
            import mediapy as media
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Writing the MIRA demo requires the FlashDreams runners extra: "
                "run `uv sync --extra runners`."
            ) from exc
        _configure_media_ffmpeg(media)
        media.write_video(str(video_path), array, fps=self.config.fps)
        stats_path = self.config.output_dir / f"stats_{self.config.runner_name}.json"
        stats_path.write_text(json.dumps(stats_history, indent=2))
        logger.info(
            f"[{self.config.runner_name}] wrote {tuple(array.shape)} -> {video_path.resolve()}"
        )
        logger.info(
            f"[{self.config.runner_name}] wrote timings -> {stats_path.resolve()}"
        )


def _configure_media_ffmpeg(media: Any) -> None:
    """Use PATH FFmpeg or imageio's bundled binary for portable MP4 output."""
    if media.video_is_available():
        return
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing the MIRA demo requires FFmpeg. Install a system FFmpeg or "
            "run `uv pip install imageio-ffmpeg`."
        ) from exc
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    if not media.video_is_available():
        raise RuntimeError("The imageio-ffmpeg executable could not be located.")


def parse_action_script(value: str) -> list[list[str]]:
    """Expand ``KEY+KEY@STEPS`` segments into per-latent held controls."""
    if not value.strip():
        raise ValueError("action_script must contain at least one segment")
    timeline: list[list[str]] = []
    for raw_segment in value.split(","):
        segment = raw_segment.strip()
        try:
            key_spec, count_spec = segment.rsplit("@", 1)
            count = int(count_spec)
        except ValueError as exc:
            raise ValueError(
                f"invalid action segment {segment!r}; expected KEY+KEY@STEPS"
            ) from exc
        if count <= 0:
            raise ValueError(f"action duration must be positive in {segment!r}")
        keys = [key.strip() for key in key_spec.split("+") if key.strip()]
        unknown = sorted(set(keys) - MIRA_KEYS)
        if unknown:
            raise ValueError(f"unknown MIRA key(s) in {segment!r}: {unknown}")
        timeline.extend([keys] * count)
    return timeline


__all__ = [
    "DEFAULT_ACTION_SCRIPT",
    "MiraDemoRunner",
    "MiraDemoRunnerConfig",
    "parse_action_script",
]
