# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-neutral construction for the T2V demo."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoAdapter
from flashdreams.serving.launch import (
    DemoDefinition,
    DemoInputMode,
)

from .backends import resolve_backend
from .runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    T2VDemoAdapter,
)

if TYPE_CHECKING:
    from .runner import T2VDemoRunnerConfig


class T2VLaunchCapability:
    """Build T2V inputs and emitted video-track defaults."""

    def adapter(self, config: RunnerConfig) -> DemoAdapter:
        return T2VDemoAdapter(
            backend=resolve_backend(cast("T2VDemoRunnerConfig", config).backend)
        )

    def demo(
        self,
        config: RunnerConfig,
        *,
        input_mode: DemoInputMode,
        scenario: Mapping[str, object],
    ) -> DemoDefinition:
        from .app import _scenario

        typed_config = cast("T2VDemoRunnerConfig", config)
        adapter = cast(T2VDemoAdapter, self.adapter(config))
        scenario = _scenario(typed_config, dict(scenario))
        preset_id = typed_config.preset_id or adapter.backend.default_preset_name
        return DemoDefinition(
            model_id=adapter.model_id,
            preset_id=preset_id,
            input_mode=input_mode,
            scenario=scenario,
            fps=int(cast(Any, scenario[FIELD_FPS])),
            video_width=int(cast(Any, scenario[FIELD_PIXEL_WIDTH])),
            video_height=int(cast(Any, scenario[FIELD_PIXEL_HEIGHT])),
            output_layout="tchw",
            config=InferenceConfig(
                model_id=adapter.model_id,
                preset_id=preset_id,
                device=typed_config.device,
                compile=typed_config.compile,
                runtime_options={"backend": adapter.backend.key},
            ),
        )


LAUNCH_CAPABILITY = T2VLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "T2VLaunchCapability"]
