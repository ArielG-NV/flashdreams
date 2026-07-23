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

import sys
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mira_integration.config import (
    load_demo_config,
)
from mira_integration.pipeline import MiraPipelineConfig
from mira_integration.scripted import parse_action_script, player_one_browser_controls
from mira_integration.webrtc.media import (
    configure_media_ffmpeg,
    normalize_player_chunk,
    tile_player_video,
)

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu


MANIFEST_PATH = (
    Path(__file__).parents[1] / "mira_integration" / "configs" / "mira_car_soccer.yaml"
)

def test_runtime_has_no_alakazam_package_imports() -> None:
    package = Path(__file__).parents[1] / "mira_integration"
    source = "\n".join(path.read_text() for path in package.glob("*.py"))
    forbidden = ("alakazam_mira", "mira_vm", "from mira ", "import mira ")
    assert not [name for name in forbidden if name in source]


def test_configure_media_ffmpeg_uses_bundled_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMedia:
        ffmpeg: str | None = None

        def video_is_available(self) -> bool:
            return self.ffmpeg is not None

        def set_ffmpeg(self, value: str) -> None:
            self.ffmpeg = value

    media = _FakeMedia()
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "bundled-ffmpeg"),
    )
    configure_media_ffmpeg(media)
    assert media.ffmpeg == "bundled-ffmpeg"


def test_parse_action_script_expands_controls() -> None:
    assert parse_action_script("W@2,W+D@1,A@2", fps=10, frames_per_chunk=1) == [
        ["W"],
        ["W"],
        ["W", "D"],
        ["A"],
        ["A"],
    ]


def test_parse_action_script_uses_100ms_duration_units() -> None:
    assert parse_action_script("W@1", fps=60, frames_per_chunk=1) == [["W"]] * 6
    assert parse_action_script("A@2", fps=30, frames_per_chunk=4) == [["A"]] * 2


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
        parse_action_script(value, fps=60, frames_per_chunk=1)
