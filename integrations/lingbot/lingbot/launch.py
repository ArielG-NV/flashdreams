# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-neutral LingBot demo construction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoAdapter
from flashdreams.serving.launch import (
    DemoDefinition,
    DemoInputMode,
)

from .demo.adapter import LingbotDemoAdapter
from .demo.spec import (
    DEFAULT_FPS,
    DEFAULT_PIXEL_HEIGHT,
    DEFAULT_PIXEL_WIDTH,
    LINGBOT_MODEL_ID,
)


class LingbotLaunchCapability:
    """Build LingBot inputs and emitted video-track defaults."""

    def adapter(self, config: RunnerConfig) -> DemoAdapter:
        del config
        return LingbotDemoAdapter()

    def demo(
        self,
        config: RunnerConfig,
        *,
        input_mode: DemoInputMode,
        scenario: Mapping[str, object],
    ) -> DemoDefinition:
        scenario = dict(scenario)
        scenario.setdefault("example_idx", getattr(config, "example_idx", 0))
        if input_mode == "replay":
            for name in (
                "prompt",
                "prompt_path",
                "image_path",
                "pose_path",
                "intrinsic_path",
                "example_data",
                "total_blocks",
                "pixel_height",
                "pixel_width",
                "fps",
            ):
                value = getattr(config, name, None)
                if value not in (None, "", ()):
                    scenario.setdefault(name, value)
        preset_id = str(getattr(config.pipeline, "name", config.runner_name))
        return DemoDefinition(
            model_id=LINGBOT_MODEL_ID,
            preset_id=preset_id,
            input_mode=input_mode,
            scenario=scenario,
            fps=int(getattr(config, "fps", DEFAULT_FPS)),
            video_width=int(getattr(config, "pixel_width", DEFAULT_PIXEL_WIDTH)),
            video_height=int(getattr(config, "pixel_height", DEFAULT_PIXEL_HEIGHT)),
            output_layout="tchw",
            config=InferenceConfig(
                model_id=LINGBOT_MODEL_ID,
                preset_id=preset_id,
                device=str(config.device),
                compile=_runner_compile(config),
                runtime_options={
                    "seed": _pipeline_seed(config),
                    "context_parallel_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "example_idx": scenario["example_idx"],
                    "total_blocks": getattr(config, "total_blocks", 1_000_000),
                },
            ),
        )


def _runner_compile(config: RunnerConfig) -> bool:
    transformer = getattr(
        getattr(config.pipeline, "diffusion_model", None),
        "transformer",
        None,
    )
    return bool(getattr(transformer, "compile_network", True))


def _pipeline_seed(config: RunnerConfig) -> int:
    diffusion_model: Any = getattr(config.pipeline, "diffusion_model", None)
    return int(getattr(diffusion_model, "seed", 42))


LAUNCH_CAPABILITY = LingbotLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "LingbotLaunchCapability"]
