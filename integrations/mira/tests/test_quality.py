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
    clear_quality_output_dir,
    find_demo_videos,
)

pytestmark = pytest.mark.ci_cpu

MANIFEST_PATH = (
    Path(__file__).parents[1] / "mira_integration" / "configs" / "mira_car_soccer.yaml"
)


def _write_demo_videos(artifacts_dir: Path) -> dict[str, Path]:
    videos = {}
    for slug in load_manifest(MANIFEST_PATH).demos:
        video = artifacts_dir / slug / "mira.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(slug.encode())
        videos[slug] = video.resolve()
    return videos


def test_find_demo_videos_requires_every_manifest_demo(tmp_path: Path) -> None:
    videos = _write_demo_videos(tmp_path)
    missing_slug = next(iter(videos))
    videos[missing_slug].unlink()

    with pytest.raises(FileNotFoundError, match=missing_slug):
        find_demo_videos(MANIFEST_PATH, tmp_path)


def test_find_demo_videos_returns_manifest_order(tmp_path: Path) -> None:
    expected = _write_demo_videos(tmp_path)

    assert find_demo_videos(MANIFEST_PATH, tmp_path) == expected


def test_clear_quality_output_only_clears_direct_child(tmp_path: Path) -> None:
    output_dir = tmp_path / "temporal-instability"
    output_dir.mkdir()
    (output_dir / "stale.csv").write_text("stale")

    resolved = clear_quality_output_dir(output_dir, artifacts_dir=tmp_path)

    assert resolved == output_dir.resolve()
    assert list(resolved.iterdir()) == []
    with pytest.raises(ValueError, match="must be a direct child"):
        clear_quality_output_dir(tmp_path, artifacts_dir=tmp_path)


def test_runner_writes_one_temporal_instability_row_per_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _write_demo_videos(tmp_path)
    output_dir = tmp_path / "temporal-instability"
    output_dir.mkdir()
    (output_dir / "stale.csv").write_text("stale")
    monkeypatch.setattr(
        "mira_integration.quality.measure_pixel_boiling",
        lambda video: float(len(video.parent.name)),
    )
    runner = MiraQualityRunner(
        MiraQualityRunnerConfig(
            runner_name="calculate-mira-quality",
            manifest=MANIFEST_PATH,
            artifacts_dir=tmp_path,
            output_dir=output_dir,
        )
    )

    runner.run()

    with (output_dir / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["runner"] for row in rows] == list(expected)
    assert all(float(row["temporal_instability_metric"]) > 0 for row in rows)
    assert not (output_dir / "stale.csv").exists()


def test_runner_does_not_clear_results_when_preflight_fails(tmp_path: Path) -> None:
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
        )
    )

    with pytest.raises(FileNotFoundError):
        runner.run()

    assert stale.read_text() == "keep"
