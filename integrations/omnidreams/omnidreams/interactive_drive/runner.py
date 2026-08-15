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

"""OmniDreams runner for the Interactive Drive application."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from flashdreams.demo import IFlashDreamsApplication, SessionInfo
from flashdreams.infra.results import StepResult
from flashdreams.runtime import StepRequirements
from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveCommand,
    InteractiveDriveRunner,
    InteractiveDriveRunnerSession,
)

from .backends.base import RenderBackend
from .config import AppConfig
from .scene_loader import load_scene_bundle
from .simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    build_ground_snapper,
    build_map_bounds,
    state_from_initial_pose,
)
from .types import (
    DriverCommand as OmnidreamsDriverCommand,
)
from .types import (
    PresentedFrame,
    SceneBundle,
)

_DEFAULT_MANIFEST = Path(__file__).parent / "configs/example_world_model.yaml"
"""Bundled production manifest used by the application entry point."""

ApplicationConfigFactory = Callable[[Sequence[str]], tuple[AppConfig, RenderBackend]]
"""Build resolved OmniDreams state from application arguments."""

SceneLoader = Callable[..., SceneBundle]
"""Load one scene bundle for an application session."""


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsInteractiveDriveSessionConfig:
    """Resolved resources for one OmniDreams driving session."""

    app: AppConfig
    """Scene, simulation, and raster settings."""

    backend: RenderBackend
    """Render backend that consumes simulated trajectory chunks."""


class OmnidreamsInteractiveDriveRunner(InteractiveDriveRunner):
    """Create isolated OmniDreams Interactive Drive runner sessions."""

    def __init__(
        self,
        *,
        config_factory: ApplicationConfigFactory | None = None,
        scene_loader: SceneLoader = load_scene_bundle,
    ) -> None:
        self._config_factory = config_factory or _application_config
        self._scene_loader = scene_loader
        self._commandline_args: tuple[str, ...] | None = None
        self._pending_config: OmnidreamsInteractiveDriveSessionConfig | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse arguments and prepare the first session's backend."""
        self._commandline_args = tuple(commandline_args)
        self._pending_config = self._create_session_config()

    def create_session(self) -> InteractiveDriveRunnerSession:
        """Create one scene-isolated OmniDreams runner session."""
        if self._commandline_args is None:
            raise RuntimeError(
                "OmnidreamsInteractiveDriveRunner.init() must run before "
                "create_session()."
            )
        config = self._pending_config
        if config is None:
            config = self._create_session_config()
        self._pending_config = None
        return OmnidreamsInteractiveDriveRunnerSession(
            config=config,
            scene_loader=self._scene_loader,
        )

    def _create_session_config(self) -> OmnidreamsInteractiveDriveSessionConfig:
        args = self._commandline_args
        if args is None:
            raise RuntimeError("Initialize the OmniDreams runner before use.")
        app_config, backend = self._config_factory(args)
        if app_config.stream_mjpeg_bind is not None:
            backend.close()
            raise ValueError(
                "--stream-mjpeg is not an application option; select a host "
                "output with --output local-window or --output webrtc."
            )
        return OmnidreamsInteractiveDriveSessionConfig(
            app=app_config,
            backend=backend,
        )


class OmnidreamsInteractiveDriveRunnerSession(InteractiveDriveRunnerSession):
    """Drive one OmniDreams scene from normalized application controls."""

    def __init__(
        self,
        *,
        config: OmnidreamsInteractiveDriveSessionConfig,
        scene_loader: SceneLoader = load_scene_bundle,
    ) -> None:
        self.config = config
        self._scene_loader = scene_loader
        self._scene: SceneBundle | None = None
        self._simulation: EgoVehicleKinematics | None = None
        self._step_index = 0
        self._closed = False

    def init(self) -> None:
        """Load the scene, warm the backend, and initialize vehicle state."""
        if self._closed:
            raise RuntimeError("Cannot initialize a closed interactive session.")
        if self._scene is not None:
            return

        app = self.config.app
        scene = self._scene_loader(
            scene_path=app.scene_path,
            camera_name=app.camera_name,
            variant=app.variant,
            prompt_override=app.prompt_override,
            raster=app.raster,
        )
        self.config.backend.warmup(scene)
        self._scene = scene
        self._simulation = EgoVehicleKinematics(
            initial_state=state_from_initial_pose(
                initial_rig_to_world=scene.initial_rig_to_world,
                initial_yaw_rad=scene.initial_yaw_rad,
                initial_speed_mps=10.0,
            ),
            vehicle_config=app.vehicle,
            ground_snapper=build_ground_snapper(scene),
            initial_timestamp_us=scene.initial_timestamp_us,
            map_bounds=build_map_bounds(scene),
            oob_margin_m=app.oob_margin_m,
            oob_warning_zone_m=app.oob_warning_zone_m,
            scene=scene,
        )

    def session_info(self) -> SessionInfo:
        """Return output geometry and presentation timing for the scene."""
        if self._scene is None:
            raise RuntimeError(
                "OmnidreamsInteractiveDriveRunnerSession.init() must run "
                "before session_info()."
            )
        app = self.config.app
        return SessionInfo(
            output_layout="tchw",
            steady_output_frame_count=app.chunk.chunk_frames,
            frames_per_second=float(app.chunk.fps),
            video_width=app.raster.width,
            video_height=app.raster.height,
            metadata={
                "scene_id": self._scene.scene_id,
                "camera_name": app.camera_name,
                "variant": app.variant,
            },
        )

    def next_step_requirements(self) -> StepRequirements | None:
        """Return the frame count required by the next simulated chunk."""
        if self._closed:
            return None
        limit = self.config.app.stop_after_consumed_chunks
        if limit is not None and self._step_index >= limit:
            return None
        frame_count = self._chunk_frame_count()
        return StepRequirements(
            step_index=self._step_index,
            input_frame_count=frame_count,
            steady_output_frame_count=self.config.app.chunk.chunk_frames,
        )

    def step(self, command: InteractiveDriveCommand) -> StepResult:
        """Simulate and render one chunk from normalized controls."""
        if self._closed:
            raise RuntimeError("Interactive driving session is closed.")
        simulation = self._simulation
        if simulation is None:
            raise RuntimeError(
                "OmnidreamsInteractiveDriveRunnerSession.init() must run before step()."
            )
        limit = self.config.app.stop_after_consumed_chunks
        if limit is not None and self._step_index >= limit:
            raise RuntimeError("Interactive driving session reached its chunk limit.")

        trajectory = simulation.pose_chunk(
            command=OmnidreamsDriverCommand(
                throttle=command.throttle,
                brake=command.brake,
                steer=command.steer,
                stop=command.stop,
                reverse=command.reverse,
                steer_is_direct=True,
                manual_control=True,
            ),
            chunk_size=self._chunk_frame_count(),
            frame_interval_s=self.config.app.chunk.frame_interval_s,
            extrapolation_offset_s=0.0,
        )
        if self._step_index == 0:
            rendered = self.config.backend.render_first_chunk(trajectory)
        else:
            rendered = self.config.backend.render_next_chunk(trajectory)
        result = StepResult.from_video_chunk(
            step_index=self._step_index,
            video_chunk=_frames_to_tchw(rendered.frames),
            layout="tchw",
            metadata={
                "source": rendered.source_name,
                "speed_mps": rendered.boundary_state_after_chunk.speed_mps,
                "actor_collision_detected": trajectory.actor_collision_detected,
            },
        )
        self._step_index += 1
        return result

    def close(self) -> None:
        """Release simulation, renderer, and model resources."""
        if self._closed:
            return
        self._closed = True
        if self._simulation is not None:
            self._simulation.close()
            self._simulation = None
        self.config.backend.close()
        self._scene = None

    def _chunk_frame_count(self) -> int:
        chunk = self.config.app.chunk
        return (
            chunk.initial_chunk_frames if self._step_index == 0 else chunk.chunk_frames
        )


def _application_config(args: Sequence[str]) -> tuple[AppConfig, RenderBackend]:
    from . import cli as _cli

    parser = _cli.build_parser()
    parser.prog = "flashdreams-run interactive-drive"
    parser.set_defaults(backend="omnidreams", manifest=_DEFAULT_MANIFEST)
    namespace = parser.parse_args(list(args))
    return _cli.prepare_config_and_backend(namespace)


def _frames_to_tchw(frames: Sequence[PresentedFrame]) -> Tensor:
    if not frames:
        raise ValueError("Interactive drive backend returned an empty frame chunk.")
    tensors = [_frame_to_chw(frame) for frame in frames]
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("Interactive drive frames must share one torch device.")
    return torch.stack(tensors, dim=0)


def _frame_to_chw(frame: PresentedFrame) -> Tensor:
    value = (
        frame.model_rgb_host_uint8
        if frame.model_rgb_host_uint8 is not None
        else frame.rgb_host_uint8
    )
    to_cuda_tensor = getattr(value, "to_cuda_tensor", None)
    if callable(to_cuda_tensor):
        value = to_cuda_tensor()
    if not isinstance(value, Tensor):
        value = torch.as_tensor(np.asarray(value))
    if value.ndim != 3:
        raise ValueError(
            "Interactive drive RGB frames must have three dimensions, "
            f"got {tuple(value.shape)}."
        )
    if value.shape[-1] == 3:
        return value.detach().permute(2, 0, 1).contiguous()
    if value.shape[0] == 3:
        return value.detach().contiguous()
    raise ValueError(
        "Interactive drive RGB frames must use HWC or CHW RGB layout, "
        f"got {tuple(value.shape)}."
    )


class OmnidreamsInteractiveDriveApplication(InteractiveDriveApplication):
    """OmniDreams Interactive Drive application."""

    def __init__(self) -> None:
        super().__init__(runner=OmnidreamsInteractiveDriveRunner())


def create_app() -> IFlashDreamsApplication:
    """Create the OmniDreams Interactive Drive application."""
    return OmnidreamsInteractiveDriveApplication()


__all__ = [
    "OmnidreamsInteractiveDriveApplication",
    "create_app",
    "OmnidreamsInteractiveDriveRunner",
    "OmnidreamsInteractiveDriveRunnerSession",
    "OmnidreamsInteractiveDriveSessionConfig",
]
