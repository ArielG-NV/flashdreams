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

"""Pandas report generation for MIRA runtime metrics."""

from __future__ import annotations

import html
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import tyro

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import Runner, RunnerConfig

_METRICS_FILENAME = "metrics_mira.csv"
"""Conventional MIRA metrics filename emitted below each demo slug."""

_SOURCE_COLUMN = "source_csv"
"""Merged-table column that identifies the CSV supplying each row."""


@dataclass(kw_only=True)
class MiraResultsViewerConfig(RunnerConfig):
    """User-facing configuration for the MIRA metrics report."""

    _target: type["MiraResultsViewer"] = field(
        default_factory=lambda: MiraResultsViewer
    )
    pipeline: Annotated[StreamInferencePipelineConfig | None, tyro.conf.Suppress] = None
    """Unused pipeline slot retained for the shared runner interface."""

    metrics_folders: Annotated[tuple[Path, ...], tyro.conf.Positional] = tyro.MISSING
    """Folders containing ``<slug>/metrics_mira.csv`` results."""

    output_dir: Path = Path("artifacts/mira-results-viewer")
    """Directory for the merged CSV, charts, and local HTML report."""

    open_browser: bool = True
    """Open the generated local report in the default browser."""


@dataclass(frozen=True)
class MiraResultsReport:
    """Paths produced by one MIRA results-viewer run."""

    csv_path: Path
    """Concatenated metrics CSV."""

    html_path: Path
    """Local HTML table and chart report."""

    fps_chart_path: Path
    """Average-FPS SVG chart."""

    p90_fps_chart_path: Path
    """90th-percentile-FPS SVG chart."""

    model_vram_chart_path: Path
    """Model-VRAM-footprint SVG chart."""


class MiraResultsViewer(Runner[MiraResultsViewerConfig, Any]):
    """Build and open a local pandas report without loading a model."""

    config: MiraResultsViewerConfig
    pipeline = None

    def __init__(self, config: MiraResultsViewerConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Concatenate MIRA metrics and open their local browser report."""
        report = build_results_report(
            self.config.metrics_folders,
            output_dir=self.config.output_dir,
        )
        report_uri = report.html_path.as_uri()
        print(f"Combined CSV: {report.csv_path}")
        print(f"Pandas HTML report: {report.html_path}")
        print(f"Rendered in your default web browser from: {report_uri}")
        print(
            "This is a local HTML file; no web server or native pandas window is used."
        )
        if self.config.open_browser:
            opened = webbrowser.open(report_uri)
            if not opened:
                print(
                    "The browser did not open automatically; open the report path above."
                )


def find_metrics_csvs(metrics_folders: tuple[Path, ...]) -> tuple[Path, ...]:
    """Find every direct-slug MIRA metrics CSV below the supplied folders.

    Args:
        metrics_folders: Parent folders whose direct children are result slugs.

    Returns:
        Files ordered by input-folder order and then slug path.

    Raises:
        ValueError: No input folders were supplied or an input is not a directory.
        FileNotFoundError: No matching metrics CSV exists.
    """
    if not metrics_folders:
        raise ValueError("Supply at least one metrics folder.")

    found: list[Path] = []
    seen: set[Path] = set()
    for folder in metrics_folders:
        resolved_folder = folder.expanduser().resolve()
        if not resolved_folder.is_dir():
            raise ValueError(f"Metrics folder is not a directory: {resolved_folder}")
        for csv_path in sorted(resolved_folder.glob(f"*/{_METRICS_FILENAME}")):
            resolved_csv = csv_path.resolve()
            if resolved_csv not in seen:
                seen.add(resolved_csv)
                found.append(resolved_csv)

    if not found:
        pattern = f"<metrics-folder>/<slug>/{_METRICS_FILENAME}"
        raise FileNotFoundError(f"No MIRA metrics files found at {pattern}.")
    return tuple(found)


def concatenate_metrics(csv_paths: tuple[Path, ...]) -> pd.DataFrame:
    """Read and concatenate MIRA metrics CSV files with source provenance.

    Args:
        csv_paths: Metrics CSV paths to read in output-row order.

    Returns:
        One dataframe containing the union of source columns.

    Raises:
        ValueError: The files contain no data rows.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame.insert(0, _SOURCE_COLUMN, str(csv_path))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("The discovered MIRA metrics CSV files contain no rows.")
    return combined


def summarize_runner_gpu_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize FPS and model VRAM for each runner and GPU configuration.

    Args:
        metrics: Concatenated MIRA metrics rows.

    Returns:
        One row per runner and GPU pair with numeric chart values.

    Raises:
        ValueError: Required columns are absent or contain no numeric values.
    """
    required = {
        "runner",
        "gpu_name",
        "runtime_median_fps",
        "runtime_p90_fps",
        "model_load_vram_gib",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"MIRA metrics are missing required columns: {missing}")

    chart_data = metrics.loc[:, sorted(required)].copy()
    chart_data["runtime_median_fps"] = pd.to_numeric(
        chart_data["runtime_median_fps"], errors="coerce"
    )
    for column in ("runtime_p90_fps", "model_load_vram_gib"):
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(
        subset=[
            "runner",
            "gpu_name",
            "runtime_median_fps",
            "runtime_p90_fps",
            "model_load_vram_gib",
        ]
    )
    if chart_data.empty:
        raise ValueError("MIRA metrics contain no complete numeric FPS and VRAM rows.")

    summary = (
        chart_data.groupby(["runner", "gpu_name"], as_index=False, sort=True)
        .agg(
            average_fps=("runtime_median_fps", "mean"),
            p90_fps=("runtime_p90_fps", "mean"),
            model_vram_footprint_gib=("model_load_vram_gib", "mean"),
        )
        .rename(
            columns={
                "runner": "Runner",
                "gpu_name": "GPU",
                "average_fps": "Average FPS",
                "p90_fps": "90th Percentile FPS",
                "model_vram_footprint_gib": "Model VRAM Footprint (GiB)",
            }
        )
    )
    summary.insert(0, "Runner + GPU", summary["Runner"] + " | " + summary["GPU"])
    return summary


def build_results_report(
    metrics_folders: tuple[Path, ...],
    *,
    output_dir: Path,
) -> MiraResultsReport:
    """Build the merged CSV, pandas charts, and local HTML report.

    Args:
        metrics_folders: Parent folders whose direct children contain metrics.
        output_dir: Destination for all generated report artifacts.

    Returns:
        Absolute paths to the generated artifacts.
    """
    csv_paths = find_metrics_csvs(metrics_folders)
    combined = concatenate_metrics(csv_paths)
    summary = summarize_runner_gpu_metrics(combined)

    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    csv_path = resolved_output / "metrics_mira_combined.csv"
    fps_chart_path = resolved_output / "average_fps_by_runner_gpu.svg"
    p90_fps_chart_path = resolved_output / "p90_fps_by_runner_gpu.svg"
    model_vram_chart_path = resolved_output / "model_vram_footprint_by_runner_gpu.svg"
    html_path = resolved_output / "mira_results.html"

    combined.to_csv(csv_path, index=False)
    _write_bar_chart(
        summary,
        value_column="Average FPS",
        title="Average FPS by Runner + GPU",
        ylabel="Average FPS",
        output_path=fps_chart_path,
        color="#76B900",
    )
    _write_bar_chart(
        summary,
        value_column="90th Percentile FPS",
        title="90th Percentile FPS by Runner + GPU",
        ylabel="90th percentile FPS",
        output_path=p90_fps_chart_path,
        color="#5B8FF9",
    )
    _write_bar_chart(
        summary,
        value_column="Model VRAM Footprint (GiB)",
        title="VRAM Footprint Of Model Config by Runner + GPU",
        ylabel="Model VRAM footprint (GiB)",
        output_path=model_vram_chart_path,
        color="#F6BD16",
    )
    html_path.write_text(
        _render_html_report(
            combined,
            summary,
            csv_path=csv_path,
            fps_chart_path=fps_chart_path,
            p90_fps_chart_path=p90_fps_chart_path,
            model_vram_chart_path=model_vram_chart_path,
        ),
        encoding="utf-8",
    )
    return MiraResultsReport(
        csv_path=csv_path,
        html_path=html_path,
        fps_chart_path=fps_chart_path,
        p90_fps_chart_path=p90_fps_chart_path,
        model_vram_chart_path=model_vram_chart_path,
    )


def _write_bar_chart(
    summary: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_data = summary.copy()
    chart_data["Runner + GPU"] = (
        chart_data["Runner"] + " | " + chart_data["GPU"].map(_chart_gpu_name)
    )
    figure_width = max(25.0, len(summary) * 1.25)
    axes = chart_data.plot.bar(
        x="Runner + GPU",
        y=value_column,
        color=color,
        figsize=(figure_width, 8),
        legend=False,
        rot=25,
    )
    axes.set_title(title)
    axes.set_xlabel("")
    axes.set_ylabel(ylabel)
    axes.grid(axis="y", alpha=0.25)
    axes.bar_label(axes.containers[0], fmt="%.2f", padding=3)
    axes.figure.tight_layout()
    axes.figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(axes.figure)


def _chart_gpu_name(gpu_name: str) -> str:
    """Return at most the first four words of a GPU name."""
    return " ".join(gpu_name.split()[:4])


def _render_html_report(
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    csv_path: Path,
    fps_chart_path: Path,
    p90_fps_chart_path: Path,
    model_vram_chart_path: Path,
) -> str:
    table = combined.to_html(
        index=False,
        border=0,
        classes="metrics-table",
        na_rep="",
        float_format=lambda value: f"{value:.3f}",
    )
    summary_table = summary.to_html(
        index=False,
        border=0,
        classes="metrics-table",
        float_format=lambda value: f"{value:.3f}",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MIRA Results Viewer</title>
  <style>
    body {{ color: #1f2937; font: 15px system-ui, sans-serif; margin: 2rem; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #5b8ff9; padding: 1rem; }}
    .table-wrap {{ max-height: 36rem; overflow: auto; }}
    .metrics-table {{ border-collapse: collapse; min-width: 100%; white-space: nowrap; }}
    .metrics-table th {{ background: #111827; color: white; position: sticky; top: 0; }}
    .metrics-table th, .metrics-table td {{ border: 1px solid #d1d5db; padding: .45rem; }}
    .metrics-table tbody tr:nth-child(even) {{ background: #f3f4f6; }}
    img {{ height: auto; max-width: 100%; }}
    code {{ overflow-wrap: anywhere; white-space: normal; }}
  </style>
</head>
<body>
  <h1>MIRA Results Viewer</h1>
  <p class="note">Pandas generated this local report. It is rendered by your
  web browser from files on this computer; no server or native pandas window
  is running.</p>
  <p>Combined CSV: <code>{html.escape(str(csv_path))}</code></p>
  <h2>Average FPS</h2>
  <img src="{html.escape(fps_chart_path.name)}" alt="Average FPS bar chart">
  <h2>90th Percentile FPS</h2>
  <img src="{html.escape(p90_fps_chart_path.name)}"
       alt="90th Percentile FPS bar chart">
  <h2>VRAM Footprint Of Model Config</h2>
  <img src="{html.escape(model_vram_chart_path.name)}"
       alt="Model VRAM footprint bar chart">
  <h2>Runner + GPU averages</h2>
  <div class="table-wrap">{summary_table}</div>
  <h2>Concatenated metrics ({len(combined)} rows)</h2>
  <div class="table-wrap">{table}</div>
</body>
</html>
"""


__all__ = [
    "MiraResultsReport",
    "MiraResultsViewer",
    "MiraResultsViewerConfig",
    "build_results_report",
    "concatenate_metrics",
    "find_metrics_csvs",
    "summarize_runner_gpu_metrics",
]
