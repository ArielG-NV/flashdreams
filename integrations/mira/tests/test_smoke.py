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

from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from flashdreams.infra.runner import RunnerConfig
from mira_integration.config import (
    MIRA_CONFIGS,
    PIPELINE_MIRA_MINI_1B,
    RUNNER_CONFIGS,
    RUNNER_MIRA_MINI_1B_DEMO,
)
from mira_integration.pipeline import DEFAULT_MODEL_REPO, MiraPipelineConfig
from mira_integration.runner import _configure_media_ffmpeg, parse_action_script

pytestmark = pytest.mark.ci_cpu


def test_static_configs_are_registered_by_matching_slug() -> None:
    assert MIRA_CONFIGS == {"mira-mini-1b-demo": PIPELINE_MIRA_MINI_1B}
    assert RUNNER_CONFIGS == {"mira-mini-1b-demo": RUNNER_MIRA_MINI_1B_DEMO}
    assert isinstance(RUNNER_MIRA_MINI_1B_DEMO, RunnerConfig)
    assert isinstance(PIPELINE_MIRA_MINI_1B, MiraPipelineConfig)
    assert RUNNER_MIRA_MINI_1B_DEMO.runner_name == PIPELINE_MIRA_MINI_1B.name
    assert PIPELINE_MIRA_MINI_1B.model_repo == DEFAULT_MODEL_REPO
    assert PIPELINE_MIRA_MINI_1B.enable_sync_and_profile


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
    _configure_media_ffmpeg(media)
    assert media.ffmpeg == "bundled-ffmpeg"


def test_parse_action_script_expands_controls() -> None:
    assert parse_action_script("W@2,W+D@1,A@2") == [
        ["W"],
        ["W"],
        ["W", "D"],
        ["A"],
        ["A"],
    ]


@pytest.mark.parametrize("value", ("", "W", "W@0", "NotAKey@1", "W@wat"))
def test_parse_action_script_rejects_invalid_input(value: str) -> None:
    with pytest.raises(ValueError):
        parse_action_script(value)


def test_entry_point_registered_when_plugin_is_installed() -> None:
    eps = {
        ep.name: ep
        for ep in entry_points(group="flashdreams.runner_configs")
        if ep.value.startswith("mira_integration.")
    }
    if not eps:
        pytest.skip("flashdreams-mira is not installed in the active environment")
    assert eps["mira-mini-1b-demo"].load() is RUNNER_MIRA_MINI_1B_DEMO
