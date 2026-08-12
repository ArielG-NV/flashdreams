# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-neutral OmniDreams demo construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoAdapter
from flashdreams.serving.launch import (
    DemoDefinition,
    DemoInputMode,
)

from .demo.adapter import OmnidreamsDemoAdapter
from .demo.spec import OMNIDREAMS_MODEL_ID


class OmnidreamsLaunchCapability:
    """Build OmniDreams inputs and emitted video-track defaults."""

    def adapter(self, config: RunnerConfig) -> DemoAdapter:
        del config
        return OmnidreamsDemoAdapter()

    def demo(
        self,
        config: RunnerConfig,
        *,
        input_mode: DemoInputMode,
        scenario: Mapping[str, object],
    ) -> DemoDefinition:
        scenario = dict(scenario)
        if input_mode == "replay":
            for name in (
                "prompt",
                "hdmap_video_paths",
                "first_frame_paths",
                "camera_names",
                "total_blocks",
                "pixel_height",
                "pixel_width",
            ):
                value = getattr(config, name, None)
                if value not in (None, "", ()):
                    scenario.setdefault(name, value)
            scenario.setdefault("fps", getattr(config, "output_fps", 30))
        preset_id = str(getattr(config.pipeline, "name", config.runner_name))
        seed = _pipeline_seed(config)
        return DemoDefinition(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=preset_id,
            input_mode=input_mode,
            scenario=scenario,
            fps=int(getattr(config, "output_fps", 30)),
            video_width=int(getattr(config, "pixel_width", 1280)),
            video_height=int(getattr(config, "pixel_height", 704)),
            config=InferenceConfig(
                model_id=OMNIDREAMS_MODEL_ID,
                preset_id=preset_id,
                device=str(config.device),
                seed=seed,
                runtime_options={
                    "pipeline_config": config.pipeline,
                    "seed": seed,
                    "release_oneshot_encoders_after_cache_init": input_mode == "replay",
                },
            ),
        )


def _pipeline_seed(config: RunnerConfig) -> int:
    diffusion_model: Any = getattr(config.pipeline, "diffusion_model", None)
    seed = getattr(diffusion_model, "seed", 42)
    return 42 if seed is None else int(seed)


LAUNCH_CAPABILITY = OmnidreamsLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "OmnidreamsLaunchCapability"]
