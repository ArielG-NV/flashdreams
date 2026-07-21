# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter and launcher for the interactive-drive manifest demo."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from test_suite_demos._adapter import ManifestDemoAdapter

_HF_HOSTS = {"huggingface.co", "www.huggingface.co"}
_DEFAULT_PORT = "18080"
_DEFAULT_STOP_AFTER_CHUNKS = "1"

_adapter = ManifestDemoAdapter(
    input_styles=("clipgt", "prompt"),
    settings=(
        "clipgt",
        "prompt",
        "perf-manifest",
        "scene-variant",
        "hf-org",
        "stream-mjpeg",
        "stop-after-chunks",
    ),
)
valid_settings = _adapter.valid_settings
valid_values = _adapter.valid_values
set_setting = _adapter.set_setting
supported_input_styles = _adapter.supported_input_styles

def build_setup_commands(*, case: dict[str, Any]) -> list[list[str]]:
    """Return the native-source preparation required by perf cases."""
    if _perf_manifest(case) is None:
        return []

    executable = shutil.which("omnidreams-prepare") or "omnidreams-prepare"
    return [[
        executable,
        "--skip-scene",
        "--skip-hf-prewarm",
        "--skip-text-encoder",
        "--perf",
    ]]


def build_command(
    *,
    case: dict[str, Any],
    output_dir: Path,
    runner_bin: str,
    no_instantiate: bool,
    materialize_assets: bool,
) -> list[str]:
    """Build the dedicated interactive-drive command for one manifest case."""
    del runner_bin
    if no_instantiate:
        raise ValueError(
            "--no-instantiate is only supported by flashdreams-run; "
            "interactive-drive must construct its scene application."
        )

    fixture = case["fixture_data"]
    source = _required_clipgt_source(fixture)
    scene = _materialize_clipgt(source) if materialize_assets else source
    executable = shutil.which("interactive-drive") or "interactive-drive"
    command = [
        executable,
        "--scene",
        str(scene),
        "--auto-start",
        "--stream-mjpeg",
        str(
            _fixture_value(
                fixture,
                "stream-mjpeg",
                default=f"127.0.0.1:{_DEFAULT_PORT}",
            )
        ),
        "--stop-after-chunks",
        str(
            _fixture_value(
                fixture,
                "stop-after-chunks",
                default=_DEFAULT_STOP_AFTER_CHUNKS,
            )
        ),
    ]
    perf_manifest = _perf_manifest(case)
    if perf_manifest is not None:
        command.extend(["--manifest", perf_manifest])
    prompt = _fixture_value(fixture, "prompt")
    if prompt is not None:
        command.extend(["--prompt", str(prompt)])
    scene_variant = _fixture_value(fixture, "scene-variant")
    if scene_variant is not None:
        command.extend(["--variant", str(scene_variant)])
    hf_org = _fixture_value(fixture, "hf-org")
    if hf_org is not None:
        command.extend(["--hf-org", str(hf_org)])
    return command


def _required_clipgt_source(fixture: dict[str, Any]) -> str:
    """Return and validate the fixture ClipGT source."""
    source = _fixture_value(fixture, "clipgt")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "interactive-drive requires fixture key clipgt with either a local "
            ".usdz path or a Hugging Face resolve URL."
        )
    return source


def _materialize_clipgt(source: str) -> Path:
    """Resolve a local archive or download one explicit Hugging Face file."""
    if source.startswith("hf://") or _is_hugging_face_url(source):
        repo_id, repo_type, revision, filename = _parse_hf_source(source)
        from huggingface_hub import hf_hub_download

        archive = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=filename,
            )
        )
    else:
        archive = Path(source).expanduser()

    if archive.suffix.lower() != ".usdz" or not archive.name.startswith("clipgt-"):
        raise ValueError(
            "interactive-drive clipgt must be a ClipGT .usdz archive, got "
            f"{archive}"
        )
    if not archive.is_file():
        raise FileNotFoundError(
            f"interactive-drive ClipGT archive does not exist: {archive}"
        )
    return archive.resolve()


def _is_hugging_face_url(source: str) -> bool:
    """Whether source is a direct huggingface.co URL."""
    return urlparse(source).netloc in _HF_HOSTS


def _parse_hf_source(source: str) -> tuple[str, str, str, str]:
    """Parse an ``hf://datasets/...`` URI or Hugging Face resolve URL."""
    parsed = urlparse(source)
    if parsed.scheme == "hf":
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        repo_type = parsed.netloc
        if repo_type not in {"datasets", "models"} or len(parts) < 4:
            raise ValueError(
                "ClipGT hf URI must be hf://datasets/<org>/<repo>/<file-path>"
            )
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        revision = parse_qs(parsed.query).get("revision", ["main"])[0]
    elif parsed.scheme in {"http", "https"} and parsed.netloc in _HF_HOSTS:
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if (
            len(parts) < 6
            or parts[0] not in {"datasets", "models"}
            or parts[3] != "resolve"
        ):
            raise ValueError(
                "ClipGT Hugging Face URL must be a direct /resolve/<revision>/ file link"
            )
        repo_type = parts[0]
        repo_id = "/".join(parts[1:3])
        revision = parts[4]
        filename = "/".join(parts[5:])
    else:
        raise ValueError(f"Unsupported ClipGT Hugging Face source: {source}")

    if not filename.startswith("scenes/"):
        raise ValueError("ClipGT Hugging Face file must be under scenes/")
    return repo_id, repo_type.rstrip("s"), revision, filename


def _fixture_value(
    fixture: dict[str, Any], key: str, *, default: Any = None
) -> Any:
    """Read one fixture value, allowing an environment-variable indirection."""
    value = fixture.get(key, default)
    return os.environ.get(value, value) if isinstance(value, str) else value


def _perf_manifest(case: dict[str, Any]) -> str | None:
    """Return the explicitly configured perf manifest for a perf case only."""
    if not case["variant"].endswith("-perf"):
        return None

    manifest = _fixture_value(case["fixture_data"], "perf-manifest")
    if manifest is None:
        return None
    if not isinstance(manifest, str) or not manifest.strip():
        raise ValueError("interactive-drive perf-manifest must be a non-empty string")
    return manifest
