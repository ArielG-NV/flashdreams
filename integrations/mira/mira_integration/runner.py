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

import asyncio
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import nvtx
import torch
import tyro
from loguru import logger

from flashdreams.core.io.disk import preflight_runtime_write_paths
from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import Runner, RunnerConfig
from mira_integration.configs.schema import (
    MiraModelMetadata,
    MiraWebRTCModelConfig,
)
from mira_integration.pipeline import MiraPipeline, MiraPipelineConfig
from mira_integration.webrtc.session import (
    MiraInferenceRuntime,
    MiraRuntimeConfig,
    normalize_player_chunk,
    tile_player_video,
)

MIRA_KEYS = frozenset(
    {"W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"}
)
DEFAULT_ACTION_SCRIPT = "W@1,W+D@1,W@1,W+A@1"
"""A short deterministic lap-like demonstration (400 ms of scripted controls)."""


@dataclass(kw_only=True)
class MiraDemoRunnerConfig(RunnerConfig):
    """User-facing configuration for the MIRA Mini example rollout."""

    _target: type["MiraDemoRunner"] = field(default_factory=lambda: MiraDemoRunner)
    pipeline: Annotated[MiraPipelineConfig, tyro.conf.Suppress]
    output_dir: Path = Path("artifacts/mira")
    """Directory for generated MIRA videos and timing data."""
    manifest: Path = tyro.MISSING
    """YAML manifest path required by ``flashdreams-run mira``."""
    demo: str = tyro.MISSING
    """Name to select from the manifest's ``demos`` mapping."""
    action_script: str = DEFAULT_ACTION_SCRIPT
    """Comma-separated ``KEY+KEY@100MS`` segments controlling player one."""
    n_diffusion_steps: int | None = None
    """Sampler steps override; ``None`` uses the selected manifest demo."""
    seed: int = 0
    """Torch RNG seed used for the autoregressive noise stream."""
    fps: int = 60
    """Output video frame rate."""

    @nvtx.annotate("MiraDemoRunnerConfig._load_selected_demo")
    def _load_selected_demo(self) -> MiraWebRTCModelConfig:
        from mira_integration.configs.manifest import load_demo_config

        return load_demo_config(self.manifest, self.demo)

    @nvtx.annotate("MiraDemoRunnerConfig.resolve")
    def resolve(self) -> MiraDemoRunnerConfig:
        """Return a copy with its manifest-selected pipeline generated."""
        selected = self._load_selected_demo()
        n_diffusion_steps = self.n_diffusion_steps
        if n_diffusion_steps is None:
            n_diffusion_steps = selected.metadata.steps
        return derive_config(
            self,
            pipeline=selected.pipeline,
            n_diffusion_steps=n_diffusion_steps,
        )


class MiraDemoRunner(Runner[MiraDemoRunnerConfig, MiraPipeline]):
    """Generate a fixed-action MIRA Mini rollout and persist MP4 + timings."""

    config: MiraDemoRunnerConfig
    pipeline: MiraPipeline | None

    def __init__(self, config: MiraDemoRunnerConfig) -> None:
        config = config.resolve()
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError(f"{config.runner_name} supports one GPU only")
        preflight_runtime_write_paths(output_dir=config.output_dir)
        self.config = config
        self.local_rank = self.global_rank = 0
        self.world_size = 1
        self.is_rank_zero = True
        self.pipeline = None

    @nvtx.annotate("MiraDemoRunner.run")
    def run(self) -> None:
        """Run the scripted demo and write a tiled MP4 plus timing JSON."""
        asyncio.run(self._run_async())

    @nvtx.annotate("MiraDemoRunner._run_async")
    async def _run_async(self) -> None:
        """Run the scripted demo through the shared MIRA runtime path."""
        selected = self.config._load_selected_demo()
        controls = parse_action_script(
            self.config.action_script,
            valid_keys=frozenset(
                binding.checkpoint_key for binding in selected.metadata.input_key_map
            ),
            fps=self.config.fps,
            frames_per_chunk=selected.metadata.frames_per_chunk,
        )
        n_diffusion_steps = self.config.n_diffusion_steps
        if n_diffusion_steps is None:
            raise RuntimeError("MIRA demo config was not resolved.")
        runtime = MiraInferenceRuntime(
            config=MiraRuntimeConfig(
                model_config=MiraWebRTCModelConfig(
                    metadata=selected.metadata,
                    pipeline=self.config.pipeline,
                ),
                device=self.config.device,
                seed=self.config.seed,
                fps=self.config.fps,
                video_height=selected.metadata.video_height,
                video_width=selected.metadata.video_width,
                frames_per_chunk=selected.metadata.frames_per_chunk,
                n_diffusion_steps=n_diffusion_steps,
                warmup_chunks=0,
            )
        )
        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float | int]] = []
        try:
            await runtime.initialize()
            await runtime.reset_for_new_session()
            for held in controls:
                with nvtx.annotate("MiraDemoRunner.run.publish_inputs"):
                    runtime.publish_player_keys(
                        _player_one_browser_controls(
                            held,
                            metadata=selected.metadata,
                        )
                    )
                with nvtx.annotate("MiraDemoRunner.run.render_frame"):
                    result = await runtime.render_next_chunk()
                chunks.append(
                    _normalize_player_chunk(
                        result.video_chunk,
                        n_players=self.config.pipeline.n_players,
                    )
                )
                if result.stats is not None:
                    stats_history.append(
                        {"autoregressive_index": result.chunk_index, **result.stats}
                    )
        finally:
            await runtime.close()

        player_video = torch.cat(chunks, dim=1)
        video = _tile_player_video(player_video).permute(0, 2, 3, 1)
        array = video.cpu().numpy()
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


@nvtx.annotate("mira.runner._player_one_controls")
def _player_one_controls(held: list[str], *, n_players: int) -> list[list[str] | None]:
    """Apply held keys to player one and leave the other players inactive."""
    if n_players <= 0:
        raise ValueError("n_players must be > 0")
    return [held] + [None] * (n_players - 1)


@nvtx.annotate("mira.runner._player_one_browser_controls")
def _player_one_browser_controls(
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


@nvtx.annotate("mira.runner._normalize_player_chunk")
def _normalize_player_chunk(output: torch.Tensor, *, n_players: int) -> torch.Tensor:
    """Return a generated chunk in ``[N,T,C,H,W]`` layout."""
    return normalize_player_chunk(output, n_players=n_players)


@nvtx.annotate("mira.runner._tile_player_video")
def _tile_player_video(video: torch.Tensor) -> torch.Tensor:
    """Tile ``[N,T,C,H,W]`` player views into ``[T,C,grid_H,grid_W]``."""
    return tile_player_video(video)


@nvtx.annotate("mira.runner._configure_media_ffmpeg")
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


@nvtx.annotate("mira.runner.parse_action_script")
def parse_action_script(
    value: str,
    *,
    valid_keys: frozenset[str] = MIRA_KEYS,
    fps: int = 60,
    frames_per_chunk: int = 1,
) -> list[list[str]]:
    """Expand ``KEY+KEY@100MS`` segments into per-chunk held controls.

    Args:
        value: Comma-separated action segments.
        valid_keys: Checkpoint keys accepted by the selected manifest demo.
        fps: Output frames per second for the scripted rollout.
        frames_per_chunk: Pixel frames emitted by one rendered MIRA chunk.

    Returns:
        Per-chunk lists of held checkpoint keys.
    """
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


__all__ = [
    "DEFAULT_ACTION_SCRIPT",
    "MiraDemoRunner",
    "MiraDemoRunnerConfig",
    "parse_action_script",
]
