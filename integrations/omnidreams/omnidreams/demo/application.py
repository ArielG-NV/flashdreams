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

"""OmniDreams replay application for the transport-neutral demo host."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from omnidreams.config import (
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
)
from omnidreams.model_session import OmnidreamsModelSessionCore
from omnidreams.runner import DEFAULT_EXAMPLE_DATA_UUID_1V, _load_video
from torch import Tensor

from flashdreams.demo import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    SessionInfo,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.results import StepResult
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    load_first_frame_tensor,
)
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepRequirements

from .spec import OmnidreamsReplayScenario, resolve_replay_scenario

PipelineFactory = Callable[[Any, str], Any]
"""Construct an initialized pipeline on the selected device."""


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsApplicationDefaults:
    """Integration-owned defaults for an OmniDreams replay application."""

    pipeline_config: Any
    """Pipeline configuration used to construct each model session."""

    prompt: str
    """Default prompt used when the replay scenario omits one."""

    total_blocks: int
    """Maximum number of autoregressive chunks generated per session."""

    pixel_height: int
    """Replay frame height in pixels."""

    pixel_width: int
    """Replay frame width in pixels."""

    fps: int
    """Presentation frame rate."""

    device: str = "cuda"
    """Device on which the pipeline is constructed."""

    seed: int | None = 42
    """Optional per-rollout diffusion seed."""

    @classmethod
    def from_runner_config(cls, runner_config: Any) -> "OmnidreamsApplicationDefaults":
        """Derive application defaults from an OmniDreams runner config."""
        required = (
            "pipeline",
            "prompt",
            "total_blocks",
            "pixel_height",
            "pixel_width",
            "output_fps",
        )
        missing = [name for name in required if not hasattr(runner_config, name)]
        if missing:
            raise TypeError(
                f"OmniDreams runner config is missing application defaults: {missing}."
            )
        diffusion_model = getattr(runner_config.pipeline, "diffusion_model", None)
        return cls(
            pipeline_config=runner_config.pipeline,
            prompt=str(runner_config.prompt),
            total_blocks=int(runner_config.total_blocks),
            pixel_height=int(runner_config.pixel_height),
            pixel_width=int(runner_config.pixel_width),
            fps=int(runner_config.output_fps),
            device=str(runner_config.device),
            seed=getattr(diffusion_model, "seed", 42),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsSessionConfig:
    """Resolved settings for one finite OmniDreams replay session."""

    pipeline_config: Any
    """Pipeline configuration selected by the application factory."""

    scenario: OmnidreamsReplayScenario
    """Validated prompts, views, and replay assets."""

    device: str
    """Device on which pipeline and conditioning tensors are allocated."""

    seed: int | None
    """Optional per-rollout diffusion seed."""

    release_oneshot_encoders_after_cache_init: bool = True
    """Whether cache initialization releases text and image encoders."""


class OmnidreamsApplication(IFlashDreamsApplication):
    """Create finite precomputed-HDMap OmniDreams replay sessions."""

    session_type: type["OmnidreamsApplicationSession"]

    def __init__(
        self,
        *,
        defaults: OmnidreamsApplicationDefaults,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.defaults = defaults
        self._pipeline_factory = pipeline_factory
        self._session_config: OmnidreamsSessionConfig | None = None

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Declare that precomputed replay consumes no live controls."""
        return CanonicalInputSchema(description="fixed OmniDreams HDMap replay")

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse replay asset, rollout, and pipeline overrides."""
        parser = _application_parser(self.defaults)
        args = parser.parse_args(list(commandline_args))
        pipeline_config = self.defaults.pipeline_config
        if args.compile is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={
                    "transformer": {"compile_network": args.compile},
                },
            )

        scenario_values: dict[str, object] = {
            "hdmap_video_paths": args.hdmap_video_paths,
            "first_frame_paths": args.first_frame_paths,
            "camera_names": args.camera_names,
            "example_data_uuid": args.example_data_uuid,
            "total_blocks": args.total_blocks,
            "pixel_height": args.pixel_height,
            "pixel_width": args.pixel_width,
            "fps": args.fps,
        }
        if args.prompt:
            scenario_values["prompt"] = args.prompt
        if args.example_data is not None:
            scenario_values["example_data"] = args.example_data
        scenario = resolve_replay_scenario(
            scenario_values,
            default_prompt=self.defaults.prompt,
        )
        self._session_config = OmnidreamsSessionConfig(
            pipeline_config=pipeline_config,
            scenario=scenario,
            device=args.device,
            seed=args.seed,
            release_oneshot_encoders_after_cache_init=(
                args.release_oneshot_encoders_after_cache_init
            ),
        )

    def create_session(self) -> IFlashDreamsApplicationSession:
        """Create one replay session with isolated autoregressive state."""
        if self._session_config is None:
            raise RuntimeError(
                "OmnidreamsApplication.init() must run before create_session()."
            )
        return self.session_type(
            config=self._session_config,
            pipeline_factory=self._pipeline_factory,
        )


class OmnidreamsApplicationSession(IFlashDreamsApplicationSession):
    """Generate one OmniDreams rollout from precomputed HDMap video."""

    def __init__(
        self,
        *,
        config: OmnidreamsSessionConfig,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.config = config
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._pipeline: Any | None = None
        self._model_session: OmnidreamsModelSessionCore | None = None
        self._hdmap_videos: Tensor | None = None
        self._frame_start = 0
        self._closed = False

    def init(self) -> None:
        """Construct the pipeline, load replay assets, and initialize its cache."""
        if self._closed:
            raise RuntimeError("Cannot initialize a closed OmniDreams session.")
        if self._pipeline is not None:
            return

        scenario = self.config.scenario
        device = torch.device(self.config.device)
        dtype = torch.bfloat16
        pipeline = self._pipeline_factory(
            self.config.pipeline_config,
            self.config.device,
        )
        self._pipeline = pipeline
        self._hdmap_videos = torch.stack(
            [
                _load_video(
                    path,
                    pixel_height=scenario.pixel_height,
                    pixel_width=scenario.pixel_width,
                    device=device,
                    dtype=dtype,
                )
                for path in scenario.hdmap_video_paths
            ],
            dim=0,
        ).unsqueeze(0)
        first_frames = [
            load_first_frame_tensor(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=device,
                dtype=dtype,
                allow_video=True,
                install_hint=DEFAULT_RUNNER_INSTALL_HINT,
            )
            for path in scenario.first_frame_paths
        ]
        model_session = OmnidreamsModelSessionCore(
            pipeline=pipeline,
            output_stream_factory=lambda: VideoOutputStream(
                postprocess_stream=None,
                output_layout="bvtchw",
            ),
        )
        self._model_session = model_session
        _seed_pipeline_for_rollout(pipeline, self.config.seed)
        model_session.reset(
            lambda: pipeline.initialize_cache(
                text=[list(scenario.prompts)],
                image=torch.stack(first_frames, dim=0).unsqueeze(0),
                view_names=list(scenario.camera_names),
            )
        )
        if self.config.release_oneshot_encoders_after_cache_init:
            release = getattr(pipeline, "release_oneshot_encoders", None)
            if callable(release):
                release()

    def session_info(self) -> SessionInfo:
        """Return multi-view output geometry and replay timing."""
        model_session = self._require_model_session()
        scenario = self.config.scenario
        steady_step_index = 1 if scenario.total_blocks > 1 else 0
        get_num_frames = getattr(self._pipeline, "get_num_frames", None)
        steady_frame_count = (
            int(get_num_frames(steady_step_index))
            if callable(get_num_frames)
            else model_session.next_num_frames()
        )
        return SessionInfo(
            output_layout="bvtchw",
            steady_output_frame_count=steady_frame_count,
            frames_per_second=float(scenario.fps),
            video_width=scenario.pixel_width,
            video_height=scenario.pixel_height,
            metadata={
                "camera_names": scenario.camera_names,
                "prompts": scenario.prompts,
            },
        )

    def next_step_requirements(self) -> StepRequirements | None:
        """Return requirements while replay frames and rollout steps remain."""
        if self._closed:
            return None
        model_session = self._require_model_session()
        scenario = self.config.scenario
        if model_session.step_index >= scenario.total_blocks:
            return None
        frame_count = model_session.next_num_frames()
        hdmaps = self._require_hdmaps()
        if self._frame_start + frame_count > hdmaps.shape[2]:
            return None
        return StepRequirements(
            step_index=model_session.step_index,
            input_frame_count=frame_count,
            steady_output_frame_count=frame_count,
        )

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        """Generate one model chunk from the next precomputed HDMap slice."""
        if inputs.values:
            raise ValueError("OmniDreams replay does not declare live inputs.")
        model_session = self._require_model_session()
        if model_session.step_index >= self.config.scenario.total_blocks:
            raise RuntimeError("OmniDreams replay has completed all configured blocks.")
        hdmaps = self._require_hdmaps()
        frame_count = model_session.next_num_frames()
        frame_end = self._frame_start + frame_count
        if frame_end > hdmaps.shape[2]:
            raise RuntimeError("OmniDreams replay HDMap input is exhausted.")
        frame_start = self._frame_start
        self._frame_start = frame_end
        result = model_session.step(
            hdmaps[:, :, frame_start:frame_end],
            metadata={
                "hdmap_frame_start": frame_start,
                "hdmap_frame_end": frame_end,
            },
        )
        return replace(result, output_window=inputs.window)

    def close(self) -> None:
        """Release replay tensors, model cache, and pipeline resources."""
        if self._closed:
            return
        self._closed = True
        if self._model_session is not None:
            self._model_session.close()
            self._model_session = None
        if self._pipeline is not None:
            close = getattr(self._pipeline, "close", None)
            if callable(close):
                close()
            self._pipeline = None
        self._hdmap_videos = None

    def _require_model_session(self) -> OmnidreamsModelSessionCore:
        if self._model_session is None:
            raise RuntimeError(
                "OmnidreamsApplicationSession.init() must run before use."
            )
        return self._model_session

    def _require_hdmaps(self) -> Tensor:
        if self._hdmap_videos is None:
            raise RuntimeError("OmniDreams replay HDMap input is not initialized.")
        return self._hdmap_videos


def _application_parser(
    defaults: OmnidreamsApplicationDefaults,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashdreams-run omnidreams")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--hdmap-video-paths", type=_split_paths, default=())
    parser.add_argument("--first-frame-paths", type=_split_paths, default=())
    parser.add_argument("--camera-names", type=_split_strings, default=())
    parser.add_argument(
        "--example-data",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--example-data-uuid",
        default=DEFAULT_EXAMPLE_DATA_UUID_1V,
    )
    parser.add_argument("--total-blocks", type=int, default=defaults.total_blocks)
    parser.add_argument("--pixel-height", type=int, default=defaults.pixel_height)
    parser.add_argument("--pixel-width", type=int, default=defaults.pixel_width)
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--release-oneshot-encoders-after-cache-init",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _split_paths(value: str) -> tuple[Path, ...]:
    return tuple(Path(part) for part in value.split(",") if part)


def _split_strings(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(",") if part)


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _seed_pipeline_for_rollout(pipeline: Any, seed: int | None) -> None:
    if seed is None:
        return
    diffusion_model = getattr(pipeline, "diffusion_model", None)
    rng = getattr(diffusion_model, "rng", None)
    if rng is not None:
        rng.manual_seed(int(seed))


def create_app() -> IFlashDreamsApplication:
    """Create the stable OmniDreams replay application."""
    return OmnidreamsApplication(
        defaults=OmnidreamsApplicationDefaults.from_runner_config(
            RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
        )
    )


def create_perf_app() -> IFlashDreamsApplication:
    """Create the optimized OmniDreams replay application."""
    return OmnidreamsApplication(
        defaults=OmnidreamsApplicationDefaults.from_runner_config(
            RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF
        )
    )


OmnidreamsApplication.session_type = OmnidreamsApplicationSession


__all__ = [
    "OmnidreamsApplication",
    "OmnidreamsApplicationDefaults",
    "OmnidreamsApplicationSession",
    "OmnidreamsSessionConfig",
    "create_app",
    "create_perf_app",
]
