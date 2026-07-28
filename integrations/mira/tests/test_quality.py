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

"""CPU tests for the MIRA quality runner orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from mira_integration.configs.manifest import load_manifest
from mira_integration.quality import (
    MiraQualityRunner,
    MiraQualityRunnerConfig,
    build_quality_render_configs,
    clear_quality_output_dir,
)
from mira_integration.runner import MiraDemoRunnerConfig

pytestmark = pytest.mark.ci_cpu

MANIFEST_PATH = (
    Path(__file__).parents[1] / "mira_integration" / "configs" / "mira_car_soccer.yaml"
)


def test_clear_quality_output_only_clears_direct_child(tmp_path: Path) -> None:
    output_dir = tmp_path / "temporal-instability"
    output_dir.mkdir()
    (output_dir / "stale.csv").write_text("stale")

    resolved = clear_quality_output_dir(output_dir, artifacts_dir=tmp_path)

    assert resolved == output_dir.resolve()
    assert list(resolved.iterdir()) == []
    with pytest.raises(ValueError, match="must be a direct child"):
        clear_quality_output_dir(tmp_path, artifacts_dir=tmp_path)


def test_quality_render_configs_use_three_trials_without_all(
    tmp_path: Path,
) -> None:
    config = MiraQualityRunnerConfig(
        runner_name="calculate-mira-quality",
        manifest=MANIFEST_PATH,
        artifacts_dir=tmp_path,
        output_dir=tmp_path / "temporal-instability",
        action_script="W@1",
        seed=17,
    )

    render_configs = build_quality_render_configs(config)

    assert list(render_configs) == list(load_manifest(MANIFEST_PATH).demos)
    assert all(len(configs) == 3 for configs in render_configs.values())
    assert all(
        render.demo != "all"
        for configs in render_configs.values()
        for render in configs
    )
    assert [render.seed for render in next(iter(render_configs.values()))] == [
        17,
        18,
        19,
    ]


def test_runner_writes_one_temporal_instability_row_per_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = load_manifest(MANIFEST_PATH).demos
    output_dir = tmp_path / "temporal-instability"
    output_dir.mkdir()
    (output_dir / "stale.csv").write_text("stale")

    class FakeDemoRunner:
        def __init__(self, config: MiraDemoRunnerConfig) -> None:
            self.config = config

        def run(self) -> None:
            config = self.config
            video = config.output_dir / config.demo / "mira.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(config.demo.encode())

    monkeypatch.setattr("mira_integration.quality.MiraDemoRunner", FakeDemoRunner)
    monkeypatch.setattr(
        "mira_integration.quality.measure_pixel_boiling",
        lambda video: float(int(video.parents[1].name.removeprefix("trial-"))),
    )
    runner = MiraQualityRunner(
        MiraQualityRunnerConfig(
            runner_name="calculate-mira-quality",
            manifest=MANIFEST_PATH,
            artifacts_dir=tmp_path,
            output_dir=output_dir,
            action_script="W@1",
        )
    )

    runner.run()

    with (output_dir / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["runner"] for row in rows] == list(expected)
    assert all(float(row["temporal_instability_metric"]) == 2.0 for row in rows)
    assert all(float(row["temporal_instability_trial_1"]) == 1.0 for row in rows)
    assert all(float(row["temporal_instability_trial_2"]) == 2.0 for row in rows)
    assert all(float(row["temporal_instability_trial_3"]) == 3.0 for row in rows)
    assert not (output_dir / "stale.csv").exists()


def test_runner_does_not_clear_results_when_action_is_invalid(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "temporal-instability"
    output_dir.mkdir()
    stale = output_dir / "results.csv"
    stale.write_text("keep")
    runner = MiraQualityRunner(
        MiraQualityRunnerConfig(
            runner_name="calculate-mira-quality",
            manifest=MANIFEST_PATH,
            artifacts_dir=tmp_path,
            output_dir=output_dir,
            action_script="UNKNOWN@1",
        )
    )

    with pytest.raises(ValueError, match="unknown MIRA key"):
        runner.run()

    assert stale.read_text() == "keep"
