# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch
from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveApplicationDefaults,
    InteractiveDriveScenarioOptions,
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


class _FakeModelSession:
    def __init__(self) -> None:
        self.inputs: list[InferenceInput] = []
        self.closed = False

    def next_step_requirements(self) -> StepRequirements | None:
        if self.inputs:
            return None
        return StepRequirements(step_index=0, input_frame_count=2)

    def step(self, inputs: InferenceInput) -> StepResult:
        self.inputs.append(inputs)
        return StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((1, 1, 2, 3, 32, 48)),
            layout="bvtchw",
        )

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
            inference_input=InferenceInput(step={"control": torch.zeros(1)})
        )

    def close(self) -> None:
        self.closed = True


def test_application_delegates_model_work_to_injected_factories() -> None:
    model_session = _FakeModelSession()
    runtime = _FakeRuntime(model_session)
    adapter = _FakeAdapter(runtime)
    providers: list[_FakeProvider] = []
    scenarios: list[InteractiveDriveScenarioOptions] = []

    def scenario_factory(
        options: InteractiveDriveScenarioOptions,
    ) -> InteractiveDriveScenarioOptions:
        scenarios.append(options)
        return options

    def provider_factory(**kwargs: Any) -> _FakeProvider:
        provider = _FakeProvider(**kwargs)
        providers.append(provider)
        return provider

    application = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(
            model_id="fake-world-model",
            preset_id="fake-preset",
            scenario_factory=scenario_factory,
            adapter_factory=lambda: adapter,
            provider_factory=provider_factory,
            scene_uuid="fake-scene",
            pixel_height=32,
            pixel_width=48,
        )
    )
    application.init(["--device", "cpu", "--total-blocks", "1"])
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
    assert adapter.config.model_id == "fake-world-model"
    assert scenarios[0].scene_uuid == "fake-scene"
    assert runtime.initial_input == InferenceInput(
        global_conditioning={"prepared": True}
    )
    assert model_session.inputs[0].step == {"control": torch.zeros(1)}
    assert result.post_processing_pipeline is not None
    assert result.post_processing_pipeline.steps[0].name == "interactive-drive-ui"
    assert torch.unique(result.video_hwc_uint8()).numel() > 1

    request, window = providers[0].windows[0]
    assert request.step_index == 0
    assert [
        (event.event_type, event.payload["key"]) for event in window.inputs.events
    ] == [("keydown", "a"), ("keydown", "w")]

    session.close()
    assert model_session.closed
    assert providers[0].closed
    assert runtime.closed


def test_application_can_disable_presentation_ui() -> None:
    model_session = _FakeModelSession()
    runtime = _FakeRuntime(model_session)
    application = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(
            model_id="fake-world-model",
            preset_id="fake-preset",
            scenario_factory=lambda options: options,
            adapter_factory=lambda: _FakeAdapter(runtime),
            provider_factory=_FakeProvider,
            scene_uuid="fake-scene",
            pixel_height=32,
            pixel_width=48,
        )
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
