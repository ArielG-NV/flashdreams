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
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import nvtx
import torch
import tyro

from flashdreams.core.io.disk import preflight_runtime_write_paths
from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.serving.webrtc.bootstrap import patch_windows_webrtc_event_loop
from mira_integration.configs.schema import MiraWebRTCModelConfig
from mira_integration.pipeline import MiraPipeline, MiraPipelineConfig
from mira_integration.scripted import (
    parse_action_script,
    run_action_script,
)
from mira_integration.webrtc.media import MiraMp4Writer
from mira_integration.webrtc.session import (
    MiraInferenceRuntime,
    MiraRuntimeConfig,
)

_ALL_DEMOS = "all"
"""Runner-only sentinel that executes every manifest demo sequentially."""


@dataclass(kw_only=True)
class MiraDemoRunnerConfig(RunnerConfig):
    """User-facing configuration for the MIRA Mini example rollout."""

    _target: type["MiraDemoRunner"] = field(default_factory=lambda: MiraDemoRunner)
    pipeline: Annotated[MiraPipelineConfig | None, tyro.conf.Suppress] = None
    """Manifest-selected pipeline generated during ``resolve``."""
    output_dir: Path = Path("artifacts/mira")
    """Directory for generated MIRA videos, latency data, and VRAM reports."""
    manifest: Path = tyro.MISSING
    """YAML manifest path required by ``flashdreams-run mira``."""
    demo: str = tyro.MISSING
    """Name to select from the manifest's ``demos`` mapping."""
    action_script: str = tyro.MISSING
    """Comma-separated ``KEY+KEY@100MS`` segments controlling player one."""
    n_diffusion_steps: int | None = None
    """Sampler steps override; ``None`` uses the selected manifest demo."""
    seed: int = 0
    """Torch RNG seed used for the autoregressive noise stream."""
    fps: int = 60
    """Output video frame rate."""
    compile_decoder: bool = True
    """Compile the stateless decoder core."""
    decoder_cuda_graph: bool = True
    """Replay the fixed-shape decoder through a CUDA graph."""
    decoder_attention_backend: Literal["torch", "triton"] = "triton"
    """Backend for short-sequence causal temporal attention."""
    decoder_warmup_chunks: int = 3
    """Unmeasured chunks used to compile, autotune, and capture the decoder."""

    @nvtx.annotate("MiraDemoRunnerConfig._load_selected_demo")
    def _load_selected_demo(self) -> MiraWebRTCModelConfig:
        from mira_integration.configs.manifest import load_demo_config

        return load_demo_config(self.manifest, self.demo)

    @nvtx.annotate("MiraDemoRunnerConfig.resolve")
    def resolve(self) -> MiraDemoRunnerConfig:
        """Return a copy with its manifest-selected pipeline generated."""
        if self.demo == _ALL_DEMOS:
            _runner_demo_names(self.manifest, self.demo)
            return derive_config(self, pipeline=None)

        selected = self._load_selected_demo()
        n_diffusion_steps = self.n_diffusion_steps
        if n_diffusion_steps is None:
            n_diffusion_steps = selected.metadata.steps
        pipeline = derive_config(
            selected.pipeline,
            enable_sync_and_profile=True,
            decoder=dict(
                compile_core=self.compile_decoder,
                use_cuda_graph=self.decoder_cuda_graph,
                causal_temporal_attention_backend=self.decoder_attention_backend,
            ),
        )
        return derive_config(
            self,
            pipeline=pipeline,
            n_diffusion_steps=n_diffusion_steps,
        )


class MiraDemoRunner(Runner[MiraDemoRunnerConfig, MiraPipeline]):
    """Generate a fixed-action MIRA Mini rollout and persist runtime artifacts."""

    config: MiraDemoRunnerConfig
    pipeline: MiraPipeline | None

    def __init__(self, config: MiraDemoRunnerConfig) -> None:
        config = config.resolve()
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError(f"{config.runner_name} supports one GPU only")
        output_dir = (
            config.output_dir
            if config.demo == _ALL_DEMOS
            else config.output_dir / config.demo
        )
        preflight_runtime_write_paths(output_dir=output_dir)
        self.config = config
        self.local_rank = self.global_rank = 0
        self.world_size = 1
        self.is_rank_zero = True
        self.pipeline = None

    @nvtx.annotate("MiraDemoRunner.run")
    def run(self) -> None:
        """Run the scripted demo and write its MP4 and runtime metrics."""
        patch_windows_webrtc_event_loop()
        asyncio.run(self._run_async())

    @nvtx.annotate("MiraDemoRunner._run_async")
    async def _run_async(self) -> None:
        """Run the scripted demo through the shared MIRA runtime path."""
        if self.config.demo == _ALL_DEMOS:
            for demo_name in _runner_demo_names(
                self.config.manifest,
                self.config.demo,
            ):
                demo_config = derive_config(
                    self.config,
                    demo=demo_name,
                    pipeline=None,
                ).resolve()
                await self._run_demo_async(demo_config)
            return
        await self._run_demo_async(self.config)

    @nvtx.annotate("MiraDemoRunner._run_demo_async")
    async def _run_demo_async(self, config: MiraDemoRunnerConfig) -> None:
        """Run one manifest-selected demo and release its runtime."""
        selected = config._load_selected_demo()
        _resolve_action_script(
            config.action_script,
            selected=selected,
            fps=config.fps,
        )
        n_diffusion_steps = config.n_diffusion_steps
        if n_diffusion_steps is None:
            raise RuntimeError("MIRA demo config was not resolved.")
        pipeline_config = self._pipeline_config(config)
        runtime = MiraInferenceRuntime(
            config=MiraRuntimeConfig(
                model_config=MiraWebRTCModelConfig(
                    metadata=selected.metadata,
                    pipeline=pipeline_config,
                ),
                device=config.device,
                seed=config.seed,
                fps=config.fps,
                n_diffusion_steps=n_diffusion_steps,
                warmup_chunks=config.decoder_warmup_chunks,
            )
        )
        try:
            await runtime.initialize()
            await runtime.reset_for_new_session()
            warmup_controls = (frozenset(),) + (None,) * (
                selected.metadata.player_count - 1
            )
            for _ in range(config.decoder_warmup_chunks):
                await runtime.generate_chunk(player_keys=warmup_controls)
            await runtime.reset_for_new_session()
            device = torch.device(config.device)
            gpu_name = (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "N/A (CPU)"
            )
            output_dir = _clear_demo_output_dir(
                config.output_dir,
                selected.metadata.name,
            )
            async with MiraMp4Writer(
                output_dir=output_dir,
                runner_name=config.runner_name,
                fps=config.fps,
                n_players=pipeline_config.n_players,
                model_vram_bytes=runtime.model_vram_footprint_bytes(),
                gpu_name=gpu_name,
                action_script=config.action_script,
            ) as writer:
                await run_action_script(
                    runtime,
                    config.action_script,
                    metadata=selected.metadata,
                    fps=config.fps,
                    on_chunk=writer.push,
                )
        finally:
            await runtime.close()

    def _pipeline_config(self, config: MiraDemoRunnerConfig) -> MiraPipelineConfig:
        """Return the manifest-resolved pipeline config."""
        pipeline = config.pipeline
        if pipeline is None:
            raise RuntimeError("MIRA demo config was not resolved.")
        return pipeline


@nvtx.annotate("mira.runner._clear_demo_output_dir")
def _clear_demo_output_dir(output_base: Path, demo_slug: str) -> Path:
    """Clear and return one concrete demo's output directory.

    Args:
        output_base: Configured parent directory for all MIRA artifacts.
        demo_slug: Concrete manifest demo name.

    Returns:
        Resolved demo output path, ready for the writer to create.

    Raises:
        ValueError: The demo path escapes ``output_base`` or is not a directory.
    """
    resolved_base = output_base.resolve()
    resolved_output = (resolved_base / demo_slug).resolve()
    if resolved_output.parent != resolved_base:
        raise ValueError(
            f"MIRA demo output must be a direct child of {resolved_base}, "
            f"got {resolved_output}."
        )
    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError(
                f"MIRA demo output exists and is not a directory: {resolved_output}."
            )
        shutil.rmtree(resolved_output)
    return resolved_output


@nvtx.annotate("mira.runner._runner_demo_names")
def _runner_demo_names(manifest_path: Path, demo: str) -> tuple[str, ...]:
    """Return concrete demo names selected by the MIRA runner.

    Args:
        manifest_path: Manifest containing the available demos.
        demo: Requested demo slug or the runner-only ``all`` sentinel.

    Returns:
        One requested slug, or every manifest slug in declaration order.
    """
    if demo != _ALL_DEMOS:
        return (demo,)
    from mira_integration.configs.manifest import load_manifest

    return tuple(load_manifest(manifest_path).demos)


@nvtx.annotate("mira.runner._resolve_action_script")
def _resolve_action_script(
    script: str,
    *,
    selected: MiraWebRTCModelConfig,
    fps: int,
) -> None:
    """Validate action script early so runner failures happen before setup."""
    parse_action_script(
        script,
        metadata=selected.metadata,
        fps=fps,
        frames_per_chunk=selected.metadata.frames_per_chunk,
    )


__all__ = [
    "MiraDemoRunner",
    "MiraDemoRunnerConfig",
    "parse_action_script",
]
