# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.scripts import cli
from flashdreams.serving import launch as launch_module
from flashdreams.serving.launch import (
    DemoDefinition,
    DemoInputMode,
    LaunchModeUnavailableError,
    LaunchOptions,
    available_launch_modes,
    resolve_launch,
)

pytestmark = pytest.mark.ci_cpu


def _runner_config(
    *,
    runner_name: str,
    pipeline_name: str | None = None,
) -> RunnerConfig:
    launch_capability = None
    if runner_name.startswith("lingbot-world"):
        launch_capability = "lingbot.launch:LAUNCH_CAPABILITY"
    elif runner_name == "omnidreams" or runner_name.startswith("omnidreams-"):
        launch_capability = "omnidreams.launch:LAUNCH_CAPABILITY"
    pipeline = SimpleNamespace(
        name=pipeline_name or runner_name,
        diffusion_model=SimpleNamespace(
            seed=42,
            transformer=SimpleNamespace(num_views=1, compile_network=True),
        ),
    )
    return cast(
        RunnerConfig,
        SimpleNamespace(
            runner_name=runner_name,
            launch_capability=launch_capability,
            pipeline=pipeline,
            device="cuda:1",
            pixel_height=480,
            pixel_width=832,
            fps=20,
            output_fps=24,
            example_idx=3,
            postprocess=SimpleNamespace(preset=""),
        ),
    )


def test_output_targets_are_inferred_from_lingbot_input_capabilities() -> None:
    assert available_launch_modes(_runner_config(runner_name="lingbot-world-fast")) == (
        "run",
        "mp4",
        "null",
        "webrtc",
        "native-window",
    )


def test_shared_mp4_target_builds_output_spec(tmp_path: Path) -> None:
    resolved = resolve_launch(
        _runner_config(runner_name="lingbot-world-fast"),
        mode="mp4",
        options=LaunchOptions(output={"path": tmp_path / "demo.mp4", "fps": 12}),
    )

    assert resolved.mode == "mp4"
    assert resolved.summary["output_path"] == tmp_path / "demo.mp4"


def test_shared_target_validates_output_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported output fields: typo"):
        resolve_launch(
            _runner_config(runner_name="lingbot-world-fast"),
            mode="webrtc",
            options=LaunchOptions(output={"typo": True}),
        )


def test_output_compatibility_does_not_depend_on_model_variant() -> None:
    config = _runner_config(
        runner_name="omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae"
    )

    assert available_launch_modes(config) == (
        "run",
        "mp4",
        "null",
        "webrtc",
        "native-window",
    )


def test_shared_webrtc_target_honors_explicit_network_precedence() -> None:
    resolved = resolve_launch(
        _runner_config(runner_name="omnidreams"),
        mode="webrtc",
        options=LaunchOptions(
            host="127.0.0.1",
            port=9011,
            output={"host": "0.0.0.0", "port": 8082},
        ),
    )

    assert resolved.summary["host"] == "127.0.0.1"
    assert resolved.summary["port"] == 9011


def test_shared_mp4_target_supplies_default_output_path() -> None:
    resolved = resolve_launch(
        _runner_config(
            runner_name="omnidreams",
            pipeline_name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae",
        ),
        mode="mp4",
    )

    assert resolved.summary == {
        "runner": "omnidreams",
        "output_target": "mp4",
        "device": "cuda:1",
        "output_path": Path("outputs/omnidreams.mp4"),
    }


class _ReplayOnlyAdapter:
    model_id = "plugin-demo"

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)


class _ReplayOnlyCapability:
    def adapter(self, config: RunnerConfig) -> Any:
        del config
        return _ReplayOnlyAdapter()

    def demo(
        self,
        config: RunnerConfig,
        *,
        input_mode: DemoInputMode,
        scenario: Mapping[str, object],
    ) -> DemoDefinition:
        del config, scenario
        return DemoDefinition(
            model_id="plugin-demo",
            input_mode=input_mode,
            config=InferenceConfig(model_id="plugin-demo"),
        )


def test_registered_targets_are_filtered_by_input_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runner_config(runner_name="third-party-model")
    config.launch_capability = "plugin:capability"
    monkeypatch.setattr(
        launch_module,
        "_load_launch_capability",
        lambda path: _ReplayOnlyCapability(),
    )

    assert available_launch_modes(config) == ("run", "mp4", "null")
    with pytest.raises(LaunchModeUnavailableError, match="not compatible"):
        resolve_launch(config, mode="webrtc")


def test_demo_capability_receives_only_input_mode_and_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Capability(_ReplayOnlyCapability):
        def demo(self, config, *, input_mode, scenario):
            captured.update(input_mode=input_mode, scenario=scenario)
            return super().demo(config, input_mode=input_mode, scenario=scenario)

    config = _runner_config(runner_name="third-party-model")
    config.launch_capability = "plugin:capability"
    monkeypatch.setattr(
        launch_module, "_load_launch_capability", lambda path: _Capability()
    )

    resolved = resolve_launch(config, mode="mp4")

    assert captured == {"input_mode": "replay", "scenario": {}}
    assert resolved.summary["output_target"] == "mp4"


def test_shared_registry_builds_webrtc_output_spec() -> None:
    resolved = resolve_launch(
        _runner_config(runner_name="lingbot-world-fast"),
        mode="webrtc",
        options=LaunchOptions(port=9000),
    )

    assert resolved.summary["port"] == 9000
    assert resolved.label == "webrtc output"


def test_no_instantiate_reports_targets_without_setting_up_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _runner_config(runner_name="lingbot-world-fast")

    cli.main(
        config,
        no_instantiate=True,
        mode="webrtc",
        host="127.0.0.1",
        port=9090,
    )

    output = capsys.readouterr().out
    assert "Available modes: run, mp4, null, webrtc, native-window" in output
    assert "Selected launch: webrtc output" in output
    assert "'host': '127.0.0.1'" in output
