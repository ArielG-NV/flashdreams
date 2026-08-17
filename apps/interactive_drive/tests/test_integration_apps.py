# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_interactive_drive_app_installs_documented_output_dependencies() -> None:
    manifest = tomllib.loads(
        (_REPO_ROOT / "apps" / "interactive_drive" / "pyproject.toml").read_text()
    )

    assert "flashdreams[local-window,serving]" in manifest["project"]["dependencies"]


def test_concrete_interactive_drive_app_is_owned_by_omnidreams() -> None:
    project_dir = _REPO_ROOT / "integrations" / "omnidreams"
    shared_app_path = _REPO_ROOT / "apps" / "interactive_drive" / "application.py"
    legacy_app_path = (
        project_dir / "omnidreams" / "interactive_drive" / "application.py"
    )
    app_path = project_dir / "omnidreams" / "interactive_drive" / "flashdreams_app.py"
    manifest = tomllib.loads((project_dir / "pyproject.toml").read_text())

    assert shared_app_path.is_file()
    assert not legacy_app_path.exists()
    assert app_path.is_file()
    assert (
        manifest["project"]["entry-points"]["flashdreams.applications"][
            "interactive-drive"
        ]
        == "omnidreams.interactive_drive.flashdreams_app:create_app"
    )
    assert "flashdreams-interactive-drive" in manifest["project"]["dependencies"]
