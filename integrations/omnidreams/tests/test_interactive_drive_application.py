# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomli
import torch
from omnidreams.interactive_drive.flashdreams_app import (
    OmnidreamsInteractiveDriveApplication,
)

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import (
    CanonicalInputWindow,
    InferenceInput,
    StepRequirements,
    StepResult,
)
from flashdreams.runtime.demo.session_inputs import PreparedStep

pytestmark = pytest.mark.ci_cpu


def test_interactive_drive_is_registered_as_application_demo() -> None:
    manifest_path = Path(__file__).parents[1] / "pyproject.toml"
    manifest = tomli.loads(manifest_path.read_text(encoding="utf-8"))

    assert (
        manifest["project"]["entry-points"]["flashdreams.applications"][
            "interactive-drive"
        ]
        == "omnidreams.interactive_drive.flashdreams_app:create_app"
    )
    assert "flashdreams-interactive-drive" in manifest["project"]["dependencies"]


class _FakeModelSession:
    def __init__(self) -> None:
        self.step_index = 0
        self.inputs: list[InferenceInput] = []
        self.closed = False

    def next_step_requirements(self) -> StepRequirements | None:
        if self.step_index >= 2:
            return None
        return StepRequirements(step_index=self.step_index, input_frame_count=2)

    def step(self, inputs: InferenceInput) -> StepResult:
        self.inputs.append(inputs)
        result = StepResult.from_video_chunk(
            step_index=self.step_index,
            video_chunk=torch.zeros((1, 1, 2, 3, 32, 48)),
            layout="bvtchw",
        )
        self.step_index += 1
        return result

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self, session: _FakeModelSession) -> None:
        self.session = session
        self.initial_input: InferenceInput | None = None
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> _FakeModelSession:
        self.initial_input = inputs
        return self.session

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.config: Any | None = None

    def create_runtime(self, config: Any) -> _FakeRuntime:
        self.config = config
        return self.runtime


class _FakeProvider:
    def __init__(self, *, scenario: Any, config: Any) -> None:
        self.scenario = scenario
        self.config = config
        self.windows: list[Any] = []
        self.closed = False

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput(global_conditioning={"prepared": True})

    def prepare_step(self, *, request: Any, user_window: Any) -> PreparedStep:
        self.windows.append((request, user_window))
        return PreparedStep(
            inference_input=InferenceInput(step={"hdmap": torch.zeros(1)})
        )

    def close(self) -> None:
        self.closed = True


def test_interactive_drive_delegates_to_omnidreams_and_registers_ui() -> None:
    model_session = _FakeModelSession()
    runtime = _FakeRuntime(model_session)
    adapter = _FakeAdapter(runtime)
    providers: list[_FakeProvider] = []

    def provider_factory(**kwargs: Any) -> _FakeProvider:
        provider = _FakeProvider(**kwargs)
        providers.append(provider)
        return provider

    application = OmnidreamsInteractiveDriveApplication(
        adapter_factory=lambda: adapter,
        provider_factory=provider_factory,
    )
    application.init(["--device", "cpu", "--total-blocks", "2"])
    session = application.create_session()
    session.init()

    result = session.step(
        CanonicalInputWindow(
            values={
                "driver_command": {
                    "throttle": 1.0,
                    "brake": 0.0,
                    "steer": 0.5,
                    "stop": False,
                    "reverse": False,
                }
            },
            metadata={"canonical_sources": {"driver_command": "gamepad"}},
            window=TimeWindow(start_s=0.0, end_s=0.1),
        )
    )

    assert adapter.config is not None
    assert adapter.config.model_id == "omnidreams"
    assert runtime.initial_input == InferenceInput(
        global_conditioning={"prepared": True}
    )
    assert model_session.inputs[0].step == {"hdmap": torch.zeros(1)}
    assert result.post_processing_chunk_index == 0
    assert result.post_processing_pipeline is not None
    assert result.post_processing_pipeline.steps[0].name == "interactive-drive-ui"
    rendered = result.video_hwc_uint8()
    assert rendered.shape == (2, 32, 48, 3)
    assert torch.unique(rendered).numel() > 1

    request, window = providers[0].windows[0]
    assert request.step_index == 0
    assert [
        (event.event_type, event.payload["key"]) for event in window.inputs.events
    ] == [
        ("keydown", "a"),
        ("keydown", "w"),
    ]
    assert window.start_s == 0.0
    assert window.end_s == pytest.approx(2 / 30)

    session.close()
    assert model_session.closed
    assert providers[0].closed
    assert runtime.closed


def test_interactive_drive_can_disable_presentation_ui() -> None:
    model_session = _FakeModelSession()
    runtime = _FakeRuntime(model_session)
    adapter = _FakeAdapter(runtime)
    application = OmnidreamsInteractiveDriveApplication(
        adapter_factory=lambda: adapter,
        provider_factory=_FakeProvider,
    )
    application.init(["--device", "cpu", "--no-ui"])
    session = application.create_session()
    session.init()

    result = session.step(
        CanonicalInputWindow(
            values={
                "driver_command": {
                    "throttle": 0.0,
                    "brake": 0.0,
                    "steer": 0.0,
                    "stop": False,
                    "reverse": False,
                }
            },
            window=TimeWindow(start_s=0.0, end_s=0.0),
        )
    )

    assert result.post_processing_pipeline is None
    session.close()
