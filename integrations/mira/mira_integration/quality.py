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
from typing import Annotated, Any

import tyro

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import Runner, RunnerConfig
from mira_integration.configs.manifest import load_manifest

_DEFAULT_MANIFEST = Path(__file__).parent / "configs" / "mira_car_soccer.yaml"
"""Packaged car-soccer manifest whose demos make up the quality suite."""

_RESULTS_FILENAME = "results.csv"
"""Machine-readable quality-suite output filename."""


@dataclass(kw_only=True)
class MiraQualityRunnerConfig(RunnerConfig):
    """User-facing configuration for MIRA video quality checks."""

    _target: type["MiraQualityRunner"] = field(
        default_factory=lambda: MiraQualityRunner
    )
    pipeline: Annotated[StreamInferencePipelineConfig | None, tyro.conf.Suppress] = None
    """Unused pipeline slot retained for the shared runner interface."""

    manifest: Path = _DEFAULT_MANIFEST
    """Manifest whose demo slugs must have generated videos."""

    artifacts_dir: Path = Path("artifacts/mira")
    """Parent directory containing one generated-artifact folder per demo."""

    output_dir: Path = Path("artifacts/mira/temporal-instability")
    """Directory replaced with the temporal-instability CSV."""


class MiraQualityRunner(Runner[MiraQualityRunnerConfig, Any]):
    """Measure quality for every generated MIRA manifest demo."""

    config: MiraQualityRunnerConfig
    pipeline = None

    def __init__(self, config: MiraQualityRunnerConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Run the quality suite after validating all required MP4 inputs."""
        videos = find_demo_videos(self.config.manifest, self.config.artifacts_dir)
        output_dir = clear_quality_output_dir(
            self.config.output_dir,
            artifacts_dir=self.config.artifacts_dir,
        )
        rows = [
            {
                "runner": runner,
                "temporal_instability_metric": measure_pixel_boiling(video),
            }
            for runner, video in videos.items()
        ]
        results_path = write_temporal_instability_results(rows, output_dir)
        print(f"MIRA temporal-instability results: {results_path}")

        # ADD NEXT TEST METHOD HERE


def find_demo_videos(manifest_path: Path, artifacts_dir: Path) -> dict[str, Path]:
    """Find one MP4 for every demo in a MIRA manifest.

    Args:
        manifest_path: YAML manifest defining the required demo slugs.
        artifacts_dir: Parent directory containing demo artifact folders.

    Returns:
        Demo slugs mapped to deterministic MP4 paths in manifest order.

    Raises:
        FileNotFoundError: Any demo folder or MP4 is missing.
    """
    manifest = load_manifest(manifest_path)
    videos: dict[str, Path] = {}
    missing: list[str] = []
    for slug in manifest.demos:
        demo_dir = artifacts_dir / slug
        candidates = sorted(demo_dir.glob("*.mp4")) if demo_dir.is_dir() else []
        if not candidates:
            missing.append(str(demo_dir / "*.mp4"))
            continue
        conventional = demo_dir / "mira.mp4"
        videos[slug] = (
            conventional if conventional in candidates else candidates[0]
        ).resolve()

    if missing:
        rendered = "\n  - ".join(missing)
        raise FileNotFoundError(
            "MIRA quality checks require an MP4 for every manifest demo. "
            f"Missing:\n  - {rendered}"
        )
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
            fieldnames=["runner", "temporal_instability_metric"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return results_path.resolve()


__all__ = [
    "MiraQualityRunner",
    "MiraQualityRunnerConfig",
    "find_demo_videos",
    "measure_pixel_boiling",
    "warp_previous_to_current",
]
