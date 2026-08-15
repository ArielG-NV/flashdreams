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

"""CPU tests for OmniDreams applications on the public demo API."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import omnidreams.demo.application as replay_module
import omnidreams.interactive_drive.runner as drive_app_module
import omnidreams.interactive_drive.runner as drive_module
import pytest
import tomli as tomllib
import torch
from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveApplicationSession,
)
from omnidreams.demo.application import (
    OmnidreamsApplication,
    OmnidreamsApplicationDefaults,
)
from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import (
    AppConfig,
    ChunkConfig,
    RasterConfig,
)
from omnidreams.interactive_drive.runner import (
    OmnidreamsInteractiveDriveRunner,
    OmnidreamsInteractiveDriveRunnerSession,
    OmnidreamsInteractiveDriveSessionConfig,
)
from omnidreams.interactive_drive.types import (
    DriverCommand,
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    VehicleState,
)

from flashdreams.demo import IFlashDreamsApplication, SessionInfo
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputWindow,
)

pytestmark = pytest.mark.ci_cpu


class _FakeDriveBackend:
    def __init__(self) -> None:
        self.warmed_scene: object | None = None
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    def warmup(self, scene: object) -> None:
        self.warmed_scene = scene

    def render_first_chunk(self, trajectory: object) -> FrameChunk:
        self.calls.append(("first", trajectory))
        return self._chunk(trajectory)

    def render_next_chunk(self, trajectory: object) -> FrameChunk:
        self.calls.append(("next", trajectory))
        return self._chunk(trajectory)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _chunk(trajectory: Any) -> FrameChunk:
        frames = tuple(
            PresentedFrame(
                timestamp_us=index,
                rgb_host_uint8=torch.full((2, 3, 3), index, dtype=torch.uint8),
                depth_host_f32=None,
            )
            for index in range(len(trajectory.timestamps_us))
        )
        return FrameChunk(
            frames=frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="fake",
        )


class _FakeSimulation:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.commands: list[DriverCommand] = []
        self.closed = False

    def pose_chunk(
        self,
        command: DriverCommand,
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> object:
        self.commands.append(command)
        assert frame_interval_s == pytest.approx(1 / 30)
        assert extrapolation_offset_s == 0.0
        state = VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=4.0,
            steer_rad=0.0,
        )
        return SimpleNamespace(
            timestamps_us=tuple(range(chunk_size)),
            boundary_state_after_chunk=state,
            actor_collision_detected=False,
        )

    def close(self) -> None:
        self.closed = True


class _FakePipeline:
    def __init__(self) -> None:
        self.diffusion_model = SimpleNamespace(
            rng=SimpleNamespace(manual_seed=lambda seed: setattr(self, "seed", seed))
        )
        self.cache_args: dict[str, object] | None = None
        self.seed: int | None = None
        self.generated_hdmaps: list[torch.Tensor] = []
        self.finalized: list[int] = []
        self.released = False
        self.closed = False

    def get_num_frames(self, step_index: int) -> int:
        del step_index
        return 2

    def initialize_cache(self, **kwargs: object) -> object:
        self.cache_args = kwargs
        return object()

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        hdmap: torch.Tensor,
    ) -> torch.Tensor:
        del cache
        self.generated_hdmaps.append(hdmap)
        return torch.full(
            (1, 1, 2, 3, 2, 2),
            autoregressive_index,
            dtype=torch.uint8,
        )

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, int]:
        del cache
        self.finalized.append(autoregressive_index)
        return {"finalize_count": 1}

    def release_oneshot_encoders(self) -> None:
        self.released = True

    def close(self) -> None:
        self.closed = True


def test_interactive_drive_application_declares_driver_command() -> None:
    backend = _FakeDriveBackend()
    app_config = AppConfig(
        scene_path=Path("scene.usdz"),
        chunk=ChunkConfig(initial_chunk_frames=2, chunk_frames=3),
        raster=RasterConfig(width=3, height=2),
        stop_after_consumed_chunks=2,
    )
    runner = OmnidreamsInteractiveDriveRunner(
        config_factory=lambda args: (app_config, cast(RenderBackend, backend)),
    )
    application = InteractiveDriveApplication(runner=runner)

    assert application.input_schema.modalities == (DRIVER_COMMAND,)
    application.init([])
    assert isinstance(application.create_session(), InteractiveDriveApplicationSession)


def test_omnidreams_runner_creates_a_backend_per_session() -> None:
    backends: list[_FakeDriveBackend] = []
    app_config = AppConfig(
        scene_path=Path("scene.usdz"),
        chunk=ChunkConfig(initial_chunk_frames=2, chunk_frames=3),
        raster=RasterConfig(width=3, height=2),
    )

    def config_factory(args: object) -> tuple[AppConfig, RenderBackend]:
        del args
        backend = _FakeDriveBackend()
        backends.append(backend)
        return app_config, cast(RenderBackend, backend)

    runner = OmnidreamsInteractiveDriveRunner(config_factory=config_factory)
    runner.init([])
    first = cast(OmnidreamsInteractiveDriveRunnerSession, runner.create_session())
    second = cast(OmnidreamsInteractiveDriveRunnerSession, runner.create_session())

    assert first.config.backend is not second.config.backend
    assert len(backends) == 2
    first.close()
    second.close()


def test_interactive_drive_session_uses_canonical_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeDriveBackend()
    scene = cast(
        SceneBundle,
        SimpleNamespace(
            scene_id="scene-1",
            initial_rig_to_world=torch.eye(4).numpy(),
            initial_yaw_rad=0.0,
            initial_timestamp_us=100,
            ground_mesh_vertices=None,
            ground_mesh_faces=None,
        ),
    )
    simulation_holder: list[_FakeSimulation] = []

    def simulation_factory(**kwargs: object) -> _FakeSimulation:
        simulation = _FakeSimulation(**kwargs)
        simulation_holder.append(simulation)
        return simulation

    monkeypatch.setattr(drive_module, "EgoVehicleKinematics", simulation_factory)
    monkeypatch.setattr(drive_module, "build_ground_snapper", lambda value: None)
    monkeypatch.setattr(drive_module, "build_map_bounds", lambda value: None)
    app_config = AppConfig(
        scene_path=Path("scene.usdz"),
        chunk=ChunkConfig(initial_chunk_frames=2, chunk_frames=3),
        raster=RasterConfig(width=3, height=2),
        stop_after_consumed_chunks=2,
    )
    runner_session = OmnidreamsInteractiveDriveRunnerSession(
        config=OmnidreamsInteractiveDriveSessionConfig(
            app=app_config, backend=cast(RenderBackend, backend)
        ),
        scene_loader=lambda **kwargs: scene,
    )
    session = InteractiveDriveApplicationSession(runner_session=runner_session)

    session.init()
    info = session.session_info()
    first_requirement = session.next_step_requirements()
    assert isinstance(info, SessionInfo)
    assert info.output_layout == "tchw"
    assert first_requirement is not None
    assert first_requirement.input_frame_count == 2
    result = session.step(_driver_inputs(throttle=0.75, steer=-0.25))

    assert result.layout == "tchw"
    assert result.video_chunk.shape == (2, 3, 2, 3)
    assert backend.calls[0][0] == "first"
    command = simulation_holder[0].commands[0]
    assert command.throttle == pytest.approx(0.75)
    assert command.steer == pytest.approx(-0.25)
    assert command.steer_is_direct is True

    second = session.next_step_requirements()
    assert second is not None
    assert second.input_frame_count == 3
    session.step(_driver_inputs())
    assert session.next_step_requirements() is None
    with pytest.raises(RuntimeError, match="chunk limit"):
        session.step(_driver_inputs())
    session.close()
    assert backend.closed is True
    assert simulation_holder[0].closed is True


def test_omnidreams_application_runs_precomputed_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdmap_path = tmp_path / "hdmap.mp4"
    frame_path = tmp_path / "first.png"
    hdmap_path.write_bytes(b"hdmap")
    frame_path.write_bytes(b"frame")
    pipeline = _FakePipeline()
    defaults = OmnidreamsApplicationDefaults(
        pipeline_config=object(),
        prompt="drive",
        total_blocks=2,
        pixel_height=2,
        pixel_width=2,
        fps=30,
        device="cpu",
        seed=7,
    )
    monkeypatch.setattr(
        replay_module,
        "_load_video",
        lambda *args, **kwargs: torch.zeros(4, 3, 2, 2),
    )
    monkeypatch.setattr(
        replay_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    application = OmnidreamsApplication(
        defaults=defaults,
        pipeline_factory=lambda config, device: pipeline,
    )
    application.init(
        [
            "--hdmap-video-paths",
            str(hdmap_path),
            "--first-frame-paths",
            str(frame_path),
            "--camera-names",
            "front",
            "--no-example-data",
        ]
    )
    session = application.create_session()

    session.init()
    assert session.session_info().output_layout == "bvtchw"
    assert pipeline.seed == 7
    assert pipeline.released is True
    first = session.step(_empty_inputs())
    second = session.step(_empty_inputs(1.0, 2.0))

    assert first.video_chunk.shape == (1, 1, 2, 3, 2, 2)
    assert second.step_index == 1
    assert session.next_step_requirements() is None
    with pytest.raises(RuntimeError, match="configured blocks"):
        session.step(_empty_inputs(2.0, 3.0))
    assert [tuple(value.shape) for value in pipeline.generated_hdmaps] == [
        (1, 1, 2, 3, 2, 2),
        (1, 1, 2, 3, 2, 2),
    ]
    session.close()
    assert pipeline.closed is True


def test_omnidreams_registers_new_application_entries() -> None:
    manifest_path = Path(__file__).parents[1] / "pyproject.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert "flashdreams-interactive-drive" in manifest["project"]["dependencies"]
    assert manifest["tool"]["uv"]["sources"]["flashdreams-interactive-drive"] == {
        "workspace": True
    }
    entries = manifest["project"]["entry-points"]["flashdreams.applications"]
    assert entries == {
        "interactive-drive": "omnidreams.interactive_drive.runner:create_app",
        "omnidreams": "omnidreams.demo.application:create_app",
        "omnidreams-perf": "omnidreams.demo.application:create_perf_app",
    }
    assert isinstance(replay_module.create_app(), IFlashDreamsApplication)
    assert isinstance(replay_module.create_perf_app(), IFlashDreamsApplication)
    assert isinstance(drive_app_module.create_app(), IFlashDreamsApplication)


def _driver_inputs(
    *,
    throttle: float = 0.0,
    brake: float = 0.0,
    steer: float = 0.0,
) -> CanonicalInputWindow:
    return CanonicalInputWindow(
        values={
            DRIVER_COMMAND.name: DRIVER_COMMAND.value(
                {
                    "throttle": throttle,
                    "brake": brake,
                    "steer": steer,
                    "stop": False,
                    "reverse": False,
                }
            )
        },
        window=TimeWindow(start_s=0.0, end_s=1.0),
    )


def _empty_inputs(start_s: float = 0.0, end_s: float = 1.0) -> CanonicalInputWindow:
    return CanonicalInputWindow(
        window=TimeWindow(start_s=start_s, end_s=end_s),
    )
