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

"""MIRA metric comparison bar-chart generation."""

from __future__ import annotations

import re
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import tyro

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import Runner, RunnerConfig

_METRICS_FILENAME = "metrics_mira.csv"
"""Conventional metrics filename below each MIRA runner slug."""

_COMPETITION_REFERENCE_PATH = (
    Path(__file__).resolve().parent / "assets" / "competition_reference.csv"
)
"""Packaged competitor measurements and their source references."""

_COMPETITOR_COLOR = "#D62728"
"""Red used for the leftmost competitor bar."""

_FLASHDREAMS_COLOR = "#2E8B57"
"""Green used for FlashDreams bars."""


@dataclass(kw_only=True)
class MiraComparisonChartGeneratorConfig(RunnerConfig):
    """User-facing configuration for one MIRA metric comparison chart."""

    _target: type["MiraComparisonChartGenerator"] = field(
        default_factory=lambda: MiraComparisonChartGenerator
    )
    pipeline: Annotated[StreamInferencePipelineConfig | None, tyro.conf.Suppress] = None
    """Unused pipeline slot retained for the shared runner interface."""

    mira_folder: Annotated[Path, tyro.conf.Positional] = tyro.MISSING
    """Folder containing one ``<runner-slug>/metrics_mira.csv`` per result."""

    metric_to_compare: str = tyro.MISSING
    """Numeric CSV column to compare."""

    runner_slug_direct_compare: str = tyro.MISSING
    """Runner slug used for the competitor and direct FlashDreams bars."""

    flashdreams_gpu_to_compare_with: str = tyro.MISSING
    """Regular expression selecting one FlashDreams GPU name."""

    competitor_gpu_to_compare_with: str = tyro.MISSING
    """Regular expression selecting one competition reference GPU name."""

    custom_y_axis: str = tyro.MISSING
    """Required y-axis label."""

    custom_title: str = tyro.MISSING
    """Required chart title."""

    flashdreams_gpu_other_runner: Annotated[
        tuple[str, ...], tyro.conf.UseAppendAction
    ] = ()
    """Additional runner slugs appended as FlashDreams bars."""

    output_path: Path = Path("artifacts/mira-comparison-chart/comparison.svg")
    """Destination SVG path."""

    open_browser: bool = True
    """Open the generated SVG in the default browser."""


@dataclass(frozen=True)
class ComparisonBar:
    """One validated bar in a MIRA comparison chart."""

    runner_slug: str
    """Label shown below the bar."""

    gpu_name: str
    """GPU represented by the bar."""

    value: float
    """Mean numeric metric value."""

    source: str
    """Result source used for color coding."""

    reference: str = ""
    """Published source URL for a competitor measurement."""


class MiraComparisonChartGenerator(Runner[MiraComparisonChartGeneratorConfig, Any]):
    """Build a focused competitor-to-FlashDreams SVG comparison."""

    config: MiraComparisonChartGeneratorConfig
    pipeline = None

    def __init__(self, config: MiraComparisonChartGeneratorConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Generate and optionally open the configured comparison chart."""
        output_path = build_comparison_chart(
            self.config.mira_folder,
            metric_to_compare=self.config.metric_to_compare,
            runner_slug_direct_compare=self.config.runner_slug_direct_compare,
            flashdreams_gpu_to_compare_with=(
                self.config.flashdreams_gpu_to_compare_with
            ),
            competitor_gpu_to_compare_with=(self.config.competitor_gpu_to_compare_with),
            custom_y_axis=self.config.custom_y_axis,
            custom_title=self.config.custom_title,
            flashdreams_gpu_other_runners=(self.config.flashdreams_gpu_other_runner),
            output_path=self.config.output_path,
        )
        print(f"Comparison chart: {output_path}")
        print(f"Rendered from: {output_path.as_uri()}")
        if self.config.open_browser and not webbrowser.open(output_path.as_uri()):
            print("The browser did not open automatically; open the chart path above.")


def read_competition_reference(
    path: Path = _COMPETITION_REFERENCE_PATH,
) -> pd.DataFrame:
    """Read and validate the competition reference CSV.

    Args:
        path: Competition reference CSV path.

    Returns:
        Reference rows with the shared MIRA metrics schema.

    Raises:
        FileNotFoundError: The reference CSV does not exist.
        ValueError: Required columns or rows are missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Competition reference CSV is missing: {path}")
    reference = pd.read_csv(path)
    required = {"runner", "gpu_name", "reference"}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(
            f"Competition reference CSV is missing required columns: {missing}"
        )
    if reference.empty:
        raise ValueError("Competition reference CSV contains no rows.")
    return reference


def collect_comparison_bars(
    mira_folder: Path,
    *,
    metric_to_compare: str,
    runner_slug_direct_compare: str,
    flashdreams_gpu_to_compare_with: str,
    competitor_gpu_to_compare_with: str,
    flashdreams_gpu_other_runners: tuple[str, ...] = (),
    competition_reference_path: Path = _COMPETITION_REFERENCE_PATH,
) -> tuple[ComparisonBar, ...]:
    """Collect validated competitor and FlashDreams bars in display order.

    Args:
        mira_folder: MIRA result root containing direct runner-slug children.
        metric_to_compare: Numeric column shared by measured and reference CSVs.
        runner_slug_direct_compare: Runner slug for the first two bars.
        flashdreams_gpu_to_compare_with: Regex selecting one measured GPU.
        competitor_gpu_to_compare_with: Regex selecting one reference GPU.
        flashdreams_gpu_other_runners: Additional measured runner slugs.
        competition_reference_path: Competition reference CSV path.

    Returns:
        Competitor first, direct FlashDreams result second, then other runners.

    Raises:
        ValueError: A slug, GPU, metric, numeric value, or regex is invalid.
    """
    resolved_folder = mira_folder.expanduser().resolve()
    if not resolved_folder.is_dir():
        raise ValueError(f"MIRA folder is not a directory: {resolved_folder}")
    try:
        gpu_pattern = re.compile(flashdreams_gpu_to_compare_with)
    except re.error as exc:
        raise ValueError(
            f"Invalid --flashdreams-gpu-to-compare-with regular expression: {exc}"
        ) from exc
    try:
        competitor_gpu_pattern = re.compile(competitor_gpu_to_compare_with)
    except re.error as exc:
        raise ValueError(
            f"Invalid --competitor-gpu-to-compare-with regular expression: {exc}"
        ) from exc

    reference = read_competition_reference(competition_reference_path)
    competitor = _select_competitor_bar(
        reference,
        runner_slug=runner_slug_direct_compare,
        gpu_pattern=competitor_gpu_pattern,
        gpu_pattern_text=competitor_gpu_to_compare_with,
        metric=metric_to_compare,
    )
    requested_runners = (
        runner_slug_direct_compare,
        *flashdreams_gpu_other_runners,
    )
    flashdreams = tuple(
        _select_flashdreams_bar(
            resolved_folder,
            runner_slug=runner_slug,
            gpu_pattern=gpu_pattern,
            gpu_pattern_text=flashdreams_gpu_to_compare_with,
            metric=metric_to_compare,
        )
        for runner_slug in requested_runners
    )
    return (competitor, *flashdreams)


def build_comparison_chart(
    mira_folder: Path,
    *,
    metric_to_compare: str,
    runner_slug_direct_compare: str,
    flashdreams_gpu_to_compare_with: str,
    competitor_gpu_to_compare_with: str,
    custom_y_axis: str,
    custom_title: str,
    flashdreams_gpu_other_runners: tuple[str, ...] = (),
    output_path: Path,
    competition_reference_path: Path = _COMPETITION_REFERENCE_PATH,
) -> Path:
    """Generate an SVG bar chart for one competitor-to-FlashDreams comparison.

    Args:
        mira_folder: MIRA result root containing direct runner-slug children.
        metric_to_compare: Numeric column shared by measured and reference CSVs.
        runner_slug_direct_compare: Runner slug for the first two bars.
        flashdreams_gpu_to_compare_with: Regex selecting one measured GPU.
        competitor_gpu_to_compare_with: Regex selecting one reference GPU.
        custom_y_axis: Text displayed on the chart's y-axis.
        custom_title: Text displayed as the chart title.
        flashdreams_gpu_other_runners: Additional measured runner slugs.
        output_path: Destination SVG path.
        competition_reference_path: Competition reference CSV path.

    Returns:
        Absolute generated SVG path.

    Raises:
        ValueError: A required presentation label is blank or comparison data
            is invalid.
    """
    if not custom_y_axis.strip():
        raise ValueError("--custom-y-axis must not be blank.")
    if not custom_title.strip():
        raise ValueError("--custom-title must not be blank.")
    bars = collect_comparison_bars(
        mira_folder,
        metric_to_compare=metric_to_compare,
        runner_slug_direct_compare=runner_slug_direct_compare,
        flashdreams_gpu_to_compare_with=flashdreams_gpu_to_compare_with,
        competitor_gpu_to_compare_with=competitor_gpu_to_compare_with,
        flashdreams_gpu_other_runners=flashdreams_gpu_other_runners,
        competition_reference_path=competition_reference_path,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    resolved_output = output_path.expanduser().resolve()
    if resolved_output.exists() and not resolved_output.is_file():
        raise ValueError(f"Comparison chart output is not a file: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(8.0, len(bars) * 1.6)
    figure_height = _comparison_figure_height(tuple(bar.runner_slug for bar in bars))
    figure = plt.figure(
        figsize=(figure_width, figure_height),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 0.18, 0.055),
        hspace=0.08,
    )
    axes = figure.add_subplot(grid[0])
    legend_axes = figure.add_subplot(grid[1])
    citation_axes = figure.add_subplot(grid[2])
    legend_axes.set_axis_off()
    citation_axes.set_axis_off()
    x_positions = list(range(len(bars)))
    colors = [
        _COMPETITOR_COLOR if bar.source == "Competitor" else _FLASHDREAMS_COLOR
        for bar in bars
    ]
    plotted = axes.bar(
        x_positions,
        [bar.value for bar in bars],
        color=colors,
    )
    axes.set_xticks(
        x_positions,
        [bar.runner_slug for bar in bars],
        rotation=20,
        ha="right",
    )
    axes.set_title(custom_title, pad=16)
    axes.set_xlabel("")
    axes.set_ylabel(custom_y_axis)
    axes.grid(axis="y", alpha=0.25)
    axes.bar_label(plotted, fmt="%.3f", padding=3)
    legend_axes.legend(
        handles=[
            Patch(
                facecolor=_COMPETITOR_COLOR,
                label=f"{_legend_gpu_name(bars[0].gpu_name)} - Competitor¹",
            ),
            Patch(
                facecolor=_FLASHDREAMS_COLOR,
                label=f"{_legend_gpu_name(bars[1].gpu_name)} - FlashDreams",
            ),
        ],
        title="GPU - Inference Platform",
        loc="center",
        ncols=1,
    )
    citation_axes.text(
        0.5,
        0.5,
        f"¹ {bars[0].reference}",
        fontsize=9,
        ha="center",
        va="center",
        transform=citation_axes.transAxes,
        url=bars[0].reference,
    )
    figure.savefig(resolved_output, format="svg", bbox_inches="tight")
    plt.close(figure)
    return resolved_output


def _comparison_figure_height(runner_slugs: tuple[str, ...]) -> float:
    """Calculate figure height from the longest rotated runner label."""
    longest_label = max((len(slug) for slug in runner_slugs), default=0)
    return max(9.5, 7.5 + longest_label * 0.05)


def _legend_gpu_name(gpu_name: str) -> str:
    """Return at most the first five words of a GPU name."""
    return " ".join(gpu_name.split()[:5])


def _select_competitor_bar(
    reference: pd.DataFrame,
    *,
    runner_slug: str,
    gpu_pattern: re.Pattern[str],
    gpu_pattern_text: str,
    metric: str,
) -> ComparisonBar:
    if metric not in reference.columns:
        raise ValueError(
            f"Metric {metric!r} is missing from competition reference CSV."
        )
    matching_runner = reference.loc[reference["runner"].astype(str) == runner_slug]
    if matching_runner.empty:
        raise ValueError(
            f"Competition reference CSV has no row for runner {runner_slug!r}."
        )
    matching = matching_runner.loc[
        matching_runner["gpu_name"]
        .astype(str)
        .map(lambda gpu: gpu_pattern.search(gpu) is not None)
    ]
    if matching.empty:
        raise ValueError(
            f"Competition runner {runner_slug!r} has no GPU matching "
            f"{gpu_pattern_text!r}."
        )
    gpu_names = matching["gpu_name"].dropna().astype(str).unique().tolist()
    if len(gpu_names) != 1:
        raise ValueError(
            f"Competitor GPU regex {gpu_pattern_text!r} matched multiple GPUs "
            f"for runner {runner_slug!r}: {gpu_names}"
        )
    gpu_name = gpu_names[0]
    value = _numeric_mean(
        matching[metric],
        description=(
            f"competition metric {metric!r} for runner {runner_slug!r} "
            f"and GPU {gpu_name!r}"
        ),
    )
    references = matching["reference"].dropna().astype(str)
    if references.empty or not references.iloc[0].strip():
        raise ValueError(f"Competition reference is missing for GPU {gpu_name!r}.")
    return ComparisonBar(
        runner_slug=runner_slug,
        gpu_name=gpu_name,
        value=value,
        source="Competitor",
        reference=references.iloc[0],
    )


def _select_flashdreams_bar(
    mira_folder: Path,
    *,
    runner_slug: str,
    gpu_pattern: re.Pattern[str],
    gpu_pattern_text: str,
    metric: str,
) -> ComparisonBar:
    metrics_path = mira_folder / runner_slug / _METRICS_FILENAME
    if not metrics_path.is_file():
        raise ValueError(
            f"Runner slug {runner_slug!r} has no metrics file at {metrics_path}."
        )
    metrics = pd.read_csv(metrics_path)
    required = {"runner", "gpu_name", metric}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(
            f"Runner slug {runner_slug!r} metrics are missing columns: {missing}"
        )
    matching_runner = metrics.loc[metrics["runner"].astype(str) == runner_slug]
    if matching_runner.empty:
        raise ValueError(
            f"Metrics file {metrics_path} has no data for runner {runner_slug!r}."
        )
    matching = matching_runner.loc[
        matching_runner["gpu_name"]
        .astype(str)
        .map(lambda gpu: gpu_pattern.search(gpu) is not None)
    ]
    if matching.empty:
        raise ValueError(
            f"Runner slug {runner_slug!r} has no GPU matching {gpu_pattern_text!r}."
        )
    gpu_names = matching["gpu_name"].dropna().astype(str).unique().tolist()
    if len(gpu_names) != 1:
        raise ValueError(
            f"GPU regex {gpu_pattern_text!r} matched multiple GPUs for runner "
            f"{runner_slug!r}: {gpu_names}"
        )
    value = _numeric_mean(
        matching[metric],
        description=f"metric {metric!r} for runner {runner_slug!r}",
    )
    return ComparisonBar(
        runner_slug=runner_slug,
        gpu_name=gpu_names[0],
        value=value,
        source="FlashDreams",
    )


def _numeric_mean(values: pd.Series, *, description: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"No numeric data exists for {description}.")
    return float(numeric.mean())


__all__ = [
    "ComparisonBar",
    "MiraComparisonChartGenerator",
    "MiraComparisonChartGeneratorConfig",
    "build_comparison_chart",
    "collect_comparison_bars",
    "read_competition_reference",
]
