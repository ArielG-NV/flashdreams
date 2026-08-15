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

"""CPU tests for the reusable Interactive Drive application."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import tomli as tomllib
import torch
from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveApplicationSession,
    InteractiveDriveCommand,
    InteractiveDriveRunner,
    InteractiveDriveRunnerSession,
)

from flashdreams.demo import SessionInfo
from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import DRIVER_COMMAND, CanonicalInputWindow, StepRequirements

pytestmark = pytest.mark.ci_cpu


class _FakeRunnerSession(InteractiveDriveRunnerSession):
    def __init__(self) -> None:
        self.initialized = False
        self.commands: list[InteractiveDriveCommand] = []
        self.closed = False

    def init(self) -> None:
        self.initialized = True

    def session_info(self) -> SessionInfo:
        return SessionInfo(
            output_layout="tchw",
            steady_output_frame_count=3,
            frames_per_second=30.0,
            video_width=4,
            video_height=2,
        )

    def next_step_requirements(self) -> StepRequirements | None:
        if self.commands:
            return None
        return StepRequirements(
            step_index=0,
            input_frame_count=2,
            steady_output_frame_count=3,
        )

    def step(self, command: InteractiveDriveCommand) -> StepResult:
        self.commands.append(command)
        return StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros(2, 3, 2, 4),
            layout="tchw",
        )

    def close(self) -> None:
        self.closed = True


class _FakeRunner(InteractiveDriveRunner):
    def __init__(self, session: _FakeRunnerSession) -> None:
        self.session = session
        self.args: tuple[str, ...] | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        self.args = tuple(commandline_args)

    def create_session(self) -> InteractiveDriveRunnerSession:
        if self.args is None:
            raise RuntimeError("runner is not initialized")
        return self.session


def test_application_wraps_runner_and_declares_driver_command() -> None:
    runner_session = _FakeRunnerSession()
    runner = _FakeRunner(runner_session)
    application = InteractiveDriveApplication(runner=runner)

    assert application.input_schema.modalities == (DRIVER_COMMAND,)
    application.init(["--scene", "scene.usdz"])
    session = application.create_session()

    assert runner.args == ("--scene", "scene.usdz")
    assert isinstance(session, InteractiveDriveApplicationSession)
    session.init()
    assert runner_session.initialized is True
    assert session.session_info().output_layout == "tchw"
    requirements = session.next_step_requirements()
    assert requirements is not None
    assert requirements.input_frame_count == 2


def test_session_normalizes_controls_and_preserves_input_window() -> None:
    runner_session = _FakeRunnerSession()
    session = InteractiveDriveApplicationSession(runner_session=runner_session)
    window = TimeWindow(start_s=1.0, end_s=2.0)

    result = session.step(
        CanonicalInputWindow(
            values={
                DRIVER_COMMAND.name: DRIVER_COMMAND.value(
                    {
                        "throttle": 0.75,
                        "brake": 0.1,
                        "steer": -0.25,
                        "stop": False,
                        "reverse": True,
                    }
                )
            },
            window=window,
        )
    )

    assert runner_session.commands == [
        InteractiveDriveCommand(
            throttle=0.75,
            brake=0.1,
            steer=-0.25,
            stop=False,
            reverse=True,
        )
    ]
    assert result.output_window == window
    session.close()
    assert runner_session.closed is True


def test_session_rejects_missing_driver_command() -> None:
    session = InteractiveDriveApplicationSession(runner_session=_FakeRunnerSession())

    with pytest.raises(TypeError, match="driver_command"):
        session.step(CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0)))


def test_package_manifest_matches_t2v_application_layout() -> None:
    manifest_path = Path(__file__).parents[1] / "pyproject.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["project"]["name"] == "flashdreams-interactive-drive"
    assert manifest["tool"]["setuptools"] == {
        "packages": ["interactive_drive"],
        "package-dir": {"interactive_drive": "."},
    }
