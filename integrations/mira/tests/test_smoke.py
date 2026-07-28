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

"""CPU-only registration and dependency-boundary checks for MIRA."""

from __future__ import annotations

import csv
from importlib.metadata import entry_points
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from mira_integration.config import load_demo_config, load_manifest
from mira_integration.pipeline import MiraPipelineConfig
from mira_integration.runner import (
    MiraDemoRunner,
    MiraDemoRunnerConfig,
    _clear_demo_output_dir,
    _runner_demo_names,
)
from mira_integration.scripted import parse_action_script, player_one_browser_controls
from mira_integration.webrtc.media import (
    _format_runtime_metrics_csv,
    _write_video,
    normalize_player_chunk,
    tile_player_video,
)

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu


MANIFEST_PATH = (
    Path(__file__).parents[1] / "mira_integration" / "configs" / "mira_car_soccer.yaml"
)
DEMO_METADATA = load_demo_config(MANIFEST_PATH, "mira-mini-1p-1b-high").metadata


def test_runtime_has_no_alakazam_package_imports() -> None:
    package = Path(__file__).parents[1] / "mira_integration"
    source = "\n".join(path.read_text() for path in package.glob("*.py"))
    forbidden = ("alakazam_mira", "mira_vm", "from mira ", "import mira ")
    assert not [name for name in forbidden if name in source]


def test_write_video_uses_pyav(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    video = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    video[1, :, :, 1] = 255

    _write_video(path, video, fps=12)

    with av.open(path) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
    assert stream.codec_context.name == "h264"
    assert stream.average_rate == 12
    assert len(decoded) == 2
    assert (decoded[0].width, decoded[0].height) == (6, 4)


def test_runtime_metrics_are_rendered_as_csv() -> None:
    rendered = _format_runtime_metrics_csv(
        runner_name="mira-mini-1p-1b-high",
        fps=60,
        model_vram_bytes=3 * 1024**3,
        gpu_name="NVIDIA Test, GPU",
        action_script="W@5,W+D@5,Space@6,W+A@5",
        stats_history=[
            {
                "total_ms": 40.0,
                "frames_per_chunk": 2,
                "mem_alloc_gib": 4.0,
            }
        ],
    )
    rows = list(csv.DictReader(rendered.splitlines()))

    assert len(rows) == 1
    assert rows[0]["runner"] == "mira-mini-1p-1b-high"
    assert rows[0]["gpu_name"] == "NVIDIA Test, GPU"
    assert rows[0]["action-script"] == "W@5,W+D@5,Space@6,W+A@5"
    assert rows[0]["frames_per_chunk"] == "2"
    assert rows[0]["runtime_latency_median_ms"] == "20.000"
    assert rows[0]["runtime_median_fps"] == "50.000"
    assert rows[0]["runtime_mem_alloc_gib"] == "4.000"


def test_runner_all_selects_every_manifest_demo() -> None:
    assert _runner_demo_names(MANIFEST_PATH, "all") == tuple(
        load_manifest(MANIFEST_PATH).demos
    )
    with pytest.raises(ValueError, match="Unknown MIRA demo 'all'"):
        load_demo_config(MANIFEST_PATH, "all")


def test_runner_clears_only_concrete_demo_output(tmp_path: Path) -> None:
    output_base = tmp_path / "mira"
    demo_output = output_base / "mira-mini-1p-1b-high"
    demo_output.mkdir(parents=True)
    (demo_output / "old.mp4").write_bytes(b"old")
    sibling = output_base / "keep"
    sibling.mkdir()
    (sibling / "marker.txt").write_text("keep")

    resolved = _clear_demo_output_dir(output_base, "mira-mini-1p-1b-high")

    assert resolved == demo_output.resolve()
    assert not demo_output.exists()
    assert (sibling / "marker.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="must be a direct child"):
        _clear_demo_output_dir(output_base, "../escaped")


@pytest.mark.asyncio
async def test_runner_all_runs_each_concrete_demo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MiraDemoRunnerConfig(
        runner_name="mira",
        manifest=MANIFEST_PATH,
        demo="all",
        action_script="W@1",
    ).resolve()
    runner = object.__new__(MiraDemoRunner)
    runner.config = config
    selected: list[str] = []

    async def record_demo(demo_config: MiraDemoRunnerConfig) -> None:
        assert demo_config.pipeline is not None
        selected.append(demo_config.demo)

    monkeypatch.setattr(runner, "_run_demo_async", record_demo)
    await runner._run_async()

    assert selected == list(load_manifest(MANIFEST_PATH).demos)
    assert "all" not in selected


def test_parse_action_script_expands_controls() -> None:
    assert parse_action_script(
        "W@2,W+D@1,A@2",
        metadata=DEMO_METADATA,
        fps=10,
        frames_per_chunk=1,
    ) == [
        ["W"],
        ["W"],
        ["W", "D"],
        ["A"],
        ["A"],
    ]


def test_parse_action_script_uses_100ms_duration_units() -> None:
    assert (
        parse_action_script(
            "W@1",
            metadata=DEMO_METADATA,
            fps=60,
            frames_per_chunk=1,
        )
        == [["W"]] * 6
    )
    assert (
        parse_action_script(
            "A@2",
            metadata=DEMO_METADATA,
            fps=30,
            frames_per_chunk=4,
        )
        == [["A"]] * 2
    )


def test_scripted_browser_controls_only_target_player_one() -> None:
    held = ["W", "D"]
    metadata = load_demo_config(MANIFEST_PATH, "mira-mini-4p").metadata
    assert player_one_browser_controls(held, metadata=metadata) == (
        frozenset({"w", "d"}),
        None,
        None,
        None,
    )


def test_scripted_video_normalizes_and_tiles_dynamic_player_count() -> None:
    single = normalize_player_chunk(torch.zeros(2, 3, 4, 5), n_players=1)
    assert single.shape == (1, 2, 3, 4, 5)

    players = torch.stack(
        tuple(torch.full((2, 3, 4, 5), float(index)) for index in range(3))
    )
    normalized = normalize_player_chunk(players, n_players=3)
    tiled = tile_player_video(normalized)
    assert normalized.shape == (3, 2, 3, 4, 5)
    assert tiled.shape == (2, 3, 8, 10)
    assert torch.equal(tiled[:, :, :4, :5], players[0])
    assert torch.equal(tiled[:, :, :4, 5:], players[1])
    assert torch.equal(tiled[:, :, 4:, :5], players[2])
    assert torch.count_nonzero(tiled[:, :, 4:, 5:]) == 0


def test_scripted_video_rejects_wrong_player_count() -> None:
    with pytest.raises(ValueError, match=r"Expected \[4,T,C,H,W\]"):
        normalize_player_chunk(torch.zeros(3, 2, 3, 4, 5), n_players=4)


@pytest.mark.parametrize("value", ("", "W", "W@0", "NotAKey@1", "W@wat"))
def test_parse_action_script_rejects_invalid_input(value: str) -> None:
    with pytest.raises(ValueError):
        parse_action_script(
            value,
            metadata=DEMO_METADATA,
            fps=60,
            frames_per_chunk=1,
        )
