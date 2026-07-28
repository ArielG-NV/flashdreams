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

"""MIRA video quality checks and CSV reporting."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any

import tyro

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import Runner, RunnerConfig
from mira_integration.configs.manifest import load_demo_config, load_manifest
from mira_integration.runner import MiraDemoRunner, MiraDemoRunnerConfig
from mira_integration.scripted import parse_action_script

_DEFAULT_MANIFEST = Path(__file__).parent / "configs" / "mira_car_soccer.yaml"
"""Packaged car-soccer manifest whose demos make up the quality suite."""

_RESULTS_FILENAME = "results.csv"
"""Machine-readable quality-suite output filename."""

_QUALITY_RENDER_COUNT = 3
"""Number of independently seeded renders scored for each manifest demo."""


@dataclass(kw_only=True)
class MiraQualityRunnerConfig(RunnerConfig):
    """User-facing configuration for MIRA video quality checks."""

    _target: type["MiraQualityRunner"] = field(
        default_factory=lambda: MiraQualityRunner
    )
    pipeline: Annotated[StreamInferencePipelineConfig | None, tyro.conf.Suppress] = None
    """Unused pipeline slot retained for the shared runner interface."""

    manifest: Path = _DEFAULT_MANIFEST
    """Manifest whose demos make up the quality suite."""

    artifacts_dir: Path = Path("artifacts/mira")
    """Parent directory containing generated quality artifacts."""

    output_dir: Path = Path("artifacts/mira/temporal-instability")
    """Directory replaced with quality renders and the result CSV."""

    action_script: str = tyro.MISSING
    """Comma-separated ``KEY+KEY@100MS`` segments used for all renders."""

    seed: int = 0
    """Base seed; the three renders use this seed and the next two."""

    fps: int = 60
    """Output video frame rate."""


class MiraQualityRunner(Runner[MiraQualityRunnerConfig, Any]):
    """Render and measure quality for every MIRA manifest demo."""

    config: MiraQualityRunnerConfig
    pipeline = None

    def __init__(self, config: MiraQualityRunnerConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Render three trials per demo and report their average score."""
        render_configs = build_quality_render_configs(self.config)
        output_dir = clear_quality_output_dir(
            self.config.output_dir,
            artifacts_dir=self.config.artifacts_dir,
        )
        videos = render_quality_videos(render_configs)
        rows: list[dict[str, str | float]] = []
        for runner, video_paths in videos.items():
            scores = [measure_pixel_boiling(video) for video in video_paths]
            row: dict[str, str | float] = {
                "runner": runner,
                "temporal_instability_metric": fmean(scores),
            }
            for trial_index, score in enumerate(scores, start=1):
                row[f"temporal_instability_trial_{trial_index}"] = score
            rows.append(row)
        results_path = write_temporal_instability_results(rows, output_dir)
        print(f"MIRA temporal-instability results: {results_path}")


def build_quality_render_configs(
    config: MiraQualityRunnerConfig,
) -> dict[str, tuple[MiraDemoRunnerConfig, ...]]:
    """Build three validated render configurations for each manifest demo."""
    if config.fps <= 0:
        raise ValueError(f"fps must be positive, got {config.fps}")

    manifest = load_manifest(config.manifest)
    render_configs: dict[str, tuple[MiraDemoRunnerConfig, ...]] = {}
    for runner_name in manifest.demos:
        selected = load_demo_config(config.manifest, runner_name)
        parse_action_script(
            config.action_script,
            metadata=selected.metadata,
            fps=config.fps,
            frames_per_chunk=selected.metadata.frames_per_chunk,
        )
        render_configs[runner_name] = tuple(
            MiraDemoRunnerConfig(
                runner_name="mira",
                manifest=config.manifest,
                demo=runner_name,
                action_script=config.action_script,
                output_dir=(
                    config.output_dir / "renders" / f"trial-{trial_index + 1}"
                ),
                device=config.device,
                seed=config.seed + trial_index,
                fps=config.fps,
            )
            for trial_index in range(_QUALITY_RENDER_COUNT)
        )
    return render_configs


def render_quality_videos(
    render_configs: dict[str, tuple[MiraDemoRunnerConfig, ...]],
) -> dict[str, tuple[Path, ...]]:
    """Render each concrete demo and return its three generated video paths."""
    videos: dict[str, tuple[Path, ...]] = {}
    for runner_name, configs in render_configs.items():
        generated: list[Path] = []
        for config in configs:
            MiraDemoRunner(config).run()
            video_path = config.output_dir.resolve() / runner_name / "mira.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"MIRA quality render did not produce {video_path}"
                )
            generated.append(video_path)
        videos[runner_name] = tuple(generated)
    return videos


def clear_quality_output_dir(output_dir: Path, *, artifacts_dir: Path) -> Path:
    """Replace the quality output directory without touching demo artifacts.

    Args:
        output_dir: Directory to clear and recreate.
        artifacts_dir: Parent MIRA artifact directory.

    Returns:
        Resolved, empty output directory.

    Raises:
        ValueError: The output is not a direct child of ``artifacts_dir``.
    """
    resolved_artifacts = artifacts_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output.parent != resolved_artifacts:
        raise ValueError(
            "MIRA quality output must be a direct child of "
            f"{resolved_artifacts}, got {resolved_output}."
        )
    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError(
                f"MIRA quality output exists and is not a directory: {resolved_output}."
            )
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True)
    return resolved_output


def warp_previous_to_current(
    previous: Any,
    current_gray: Any,
    previous_gray: Any,
) -> Any:
    """Warp a previous frame into the current frame coordinate system.

    Args:
        previous: Previous grayscale frame to warp.
        current_gray: Current grayscale frame used for optical flow.
        previous_gray: Previous grayscale frame used for optical flow.

    Returns:
        Motion-compensated previous frame.
    """
    import cv2
    import numpy as np

    initial_flow: Any = None
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        initial_flow,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    height, width = current_gray.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        previous,
        x + flow[..., 0],
        y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def measure_pixel_boiling(video_path: Path) -> float:
    """Measure motion-compensated temporal instability in one video.

    Args:
        video_path: MP4 containing at least two readable frames.

    Returns:
        Mean frame-transition luminance residual on the ``0`` to ``255`` scale.

    Raises:
        ValueError: The video cannot be opened or has fewer than two frames.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    try:
        ok, previous = capture.read()
        if not ok:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        frame_scores: list[float] = []

        while True:
            ok, current = capture.read()
            if not ok:
                break
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            previous_smooth = cv2.GaussianBlur(previous_gray, (3, 3), 0)
            current_smooth = cv2.GaussianBlur(current_gray, (3, 3), 0)
            warped_previous = warp_previous_to_current(
                previous_smooth,
                current_smooth,
                previous_smooth,
            )
            residual = cv2.absdiff(current_smooth, warped_previous).astype(np.float32)

            border = 10
            if residual.shape[0] > 2 * border and residual.shape[1] > 2 * border:
                residual = residual[border:-border, border:-border]
            frame_scores.append(float(np.mean(residual)))
            previous_gray = current_gray
    finally:
        capture.release()

    if not frame_scores:
        raise ValueError(
            f"Video must contain at least two readable frames: {video_path}"
        )
    return float(np.mean(frame_scores))


def write_temporal_instability_results(
    rows: list[dict[str, str | float]],
    output_dir: Path,
) -> Path:
    """Write temporal-instability rows to the suite CSV.

    Args:
        rows: Per-runner quality measurements.
        output_dir: Existing destination directory.

    Returns:
        Resolved ``results.csv`` path.
    """
    results_path = output_dir / _RESULTS_FILENAME
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "runner",
                "temporal_instability_metric",
                *[
                    f"temporal_instability_trial_{trial_index}"
                    for trial_index in range(1, _QUALITY_RENDER_COUNT + 1)
                ],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return results_path.resolve()


__all__ = [
    "MiraQualityRunner",
    "MiraQualityRunnerConfig",
    "build_quality_render_configs",
    "clear_quality_output_dir",
    "measure_pixel_boiling",
    "render_quality_videos",
    "warp_previous_to_current",
    "write_temporal_instability_results",
]
