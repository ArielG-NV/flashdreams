# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable interactive driving application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from omnidreams.demo.adapter import OmnidreamsDemoAdapter
from omnidreams.demo.providers import LudusSceneConditioningProvider
from omnidreams.demo.spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    OMNIDREAMS_MODEL_ID,
    LudusBackendName,
    OmnidreamsLudusReplayScenario,
)
from omnidreams.runner import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH

from flashdreams.demo import IFlashDreamsApplication
from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveApplicationDefaults,
    InteractiveDriveApplicationSession,
    InteractiveDriveScenarioOptions,
)

AdapterFactory = Callable[[], Any]
ProviderFactory = Callable[..., Any]


def _create_omnidreams_scenario(
    options: InteractiveDriveScenarioOptions,
) -> OmnidreamsLudusReplayScenario:
    """Create an OmniDreams scenario from shared interactive-drive settings."""
    return OmnidreamsLudusReplayScenario(
        keyboard_events=(),
        scene_path=options.scene_path,
        scene_dir=options.scene_dir,
        scene_uuid=options.scene_uuid,
        scene_variant=options.scene_variant,
        camera_name=options.camera_name,
        prompt=options.prompt,
        total_blocks=options.total_blocks,
        pixel_height=options.pixel_height,
        pixel_width=options.pixel_width,
        fps=options.fps,
        move_speed_per_s=options.move_speed_per_s,
        rotate_speed_rad_per_s=options.rotate_speed_rad_per_s,
        ludus_backend=cast(LudusBackendName, options.scene_backend),
    )


OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS = InteractiveDriveApplicationDefaults(
    model_id=OMNIDREAMS_MODEL_ID,
    preset_id=DEFAULT_OMNIDREAMS_PRESET,
    scenario_factory=_create_omnidreams_scenario,
    adapter_factory=OmnidreamsDemoAdapter,
    provider_factory=LudusSceneConditioningProvider,
    scene_uuid=DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    pixel_height=DEFAULT_VIDEO_HEIGHT,
    pixel_width=DEFAULT_VIDEO_WIDTH,
    scene_backend_aliases=("--ludus-backend",),
    ui_title="OMNIDREAMS / INTERACTIVE DRIVE",
)
"""Runtime defaults for the OmniDreams interactive driving application."""


class OmnidreamsInteractiveDriveApplication(InteractiveDriveApplication):
    """Interactive driving application backed by OmniDreams."""

    session_type = InteractiveDriveApplicationSession

    def __init__(
        self,
        *,
        adapter_factory: AdapterFactory = OmnidreamsDemoAdapter,
        provider_factory: ProviderFactory = LudusSceneConditioningProvider,
    ) -> None:
        super().__init__(
            defaults=replace(
                OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
                adapter_factory=adapter_factory,
                provider_factory=provider_factory,
            )
        )


def create_app() -> IFlashDreamsApplication:
    """Create the OmniDreams interactive driving application."""
    return OmnidreamsInteractiveDriveApplication()


__all__ = [
    "OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS",
    "OmnidreamsInteractiveDriveApplication",
    "create_app",
]
