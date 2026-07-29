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
import shutil
import subprocess
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

_TEMPORAL_INSTABILITY_RESULTS_PATH = Path("temporal-instability/results.csv")
"""Quality-suite results path relative to a MIRA artifact folder."""

_SOURCE_COLUMN = "source_csv"
"""Merged-table column that identifies the CSV supplying each row."""

_GPU_BAR_COLORS = (
    "#176B3A",
    "#86D17A",
    "#2E8B57",
    "#B7E4C7",
    "#52B788",
    "#D8F3DC",
)
"""Green chart palette ordered from the first discovered GPU onward."""

_COMPETITION_REFERENCE_PATH = (
    Path(__file__).resolve().parent / "assets" / "competition_reference.csv"
)
"""Packaged competitor measurements included in the average-FPS chart."""

_ALAKAZAM_BAR_COLOR = "#FF0000"
"""Bar color used to distinguish Alakazam reference results."""


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

    temporal_instability_mira_folder: Path = tyro.MISSING
    """MIRA folder containing the required ``temporal-instability/results.csv``."""

    ignore_runner_slug: tuple[str, ...] = ()
    """Runner slugs to omit from every generated report artifact."""

    output_dir: Path = Path("artifacts/mira-results-viewer")
    """Directory for the generated CSV files and local HTML report."""

    open_browser: bool = True
    """Open the generated local report in the default browser."""


@dataclass(frozen=True)
class MiraResultsReport:
    """Paths produced by one MIRA results-viewer run."""

    csv_path: Path
    """Concatenated metrics CSV."""

    matrix_csv_path: Path
    """Runner-by-GPU average-FPS and quality CSV."""

    html_path: Path
    """Local runner performance and quality report."""


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
            temporal_instability_mira_folder=self.config.temporal_instability_mira_folder,
            ignore_runner_slugs=self.config.ignore_runner_slug,
        )
        report_uri = report.html_path.as_uri()
        print(f"Combined CSV: {report.csv_path}")
        print(f"Runner/GPU/quality CSV: {report.matrix_csv_path}")
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
        "runtime_average_fps",
        "runtime_1_percent_lows_fps",
        "model_load_vram_gib",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"MIRA metrics are missing required columns: {missing}")

    chart_data = metrics.loc[:, sorted(required)].copy()
    chart_data["runtime_average_fps"] = pd.to_numeric(
        chart_data["runtime_average_fps"], errors="coerce"
    )
    for column in ("runtime_1_percent_lows_fps", "model_load_vram_gib"):
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(
        subset=[
            "runner",
            "gpu_name",
            "runtime_average_fps",
            "runtime_1_percent_lows_fps",
            "model_load_vram_gib",
        ]
    )
    if chart_data.empty:
        raise ValueError("MIRA metrics contain no complete numeric FPS and VRAM rows.")

    summary = (
        chart_data.groupby(["runner", "gpu_name"], as_index=False, sort=True)
        .agg(
            average_fps=("runtime_average_fps", "mean"),
            one_percent_lows_fps=("runtime_1_percent_lows_fps", "mean"),
            model_vram_footprint_gib=("model_load_vram_gib", "mean"),
        )
        .rename(
            columns={
                "runner": "Runner",
                "gpu_name": "GPU",
                "average_fps": "Average FPS",
                "one_percent_lows_fps": "Average FPS 1% Lows",
                "model_vram_footprint_gib": "Model VRAM Footprint (GiB)",
            }
        )
    )
    summary.insert(0, "Runner + GPU", summary["Runner"] + " | " + summary["GPU"])
    return summary


def read_temporal_instability_metrics(mira_folder: Path) -> pd.DataFrame:
    """Read and validate temporal-instability metrics below a MIRA folder.

    Args:
        mira_folder: MIRA artifact root containing quality-suite results.

    Returns:
        One row per runner with a numeric temporal-instability metric.

    Raises:
        ValueError: The MIRA folder or required columns are invalid.
        FileNotFoundError: The quality-suite results CSV is missing.
    """
    resolved_folder = mira_folder.expanduser().resolve()
    if not resolved_folder.is_dir():
        raise ValueError(
            f"Temporal-instability MIRA folder is not a directory: {resolved_folder}"
        )

    results_path = resolved_folder / _TEMPORAL_INSTABILITY_RESULTS_PATH
    if not results_path.is_file():
        raise FileNotFoundError(
            f"No temporal-instability results found at {results_path}."
        )

    metrics = pd.read_csv(results_path)
    required = {"runner", "temporal_instability_metric"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(
            f"Temporal-instability results are missing required columns: {missing}"
        )

    chart_data = metrics.loc[:, ["runner", "temporal_instability_metric"]].copy()
    chart_data["temporal_instability_metric"] = pd.to_numeric(
        chart_data["temporal_instability_metric"],
        errors="coerce",
    )
    chart_data = chart_data.dropna(subset=["runner", "temporal_instability_metric"])
    if chart_data.empty:
        raise ValueError("Temporal-instability results contain no numeric metric rows.")

    summary = (
        chart_data.groupby("runner", as_index=False, sort=True)
        .agg(
            temporal_instability_metric=(
                "temporal_instability_metric",
                "mean",
            )
        )
        .rename(
            columns={
                "runner": "Runner",
                "temporal_instability_metric": "Temporal Instability",
            }
        )
    )
    return summary


def build_results_report(
    metrics_folders: tuple[Path, ...],
    *,
    output_dir: Path,
    temporal_instability_mira_folder: Path,
    ignore_runner_slugs: tuple[str, ...] = (),
) -> MiraResultsReport:
    """Build the merged CSV, pandas charts, and local HTML report.

    Args:
        metrics_folders: Parent folders whose direct children contain metrics.
        output_dir: Destination for all generated report artifacts.
        temporal_instability_mira_folder: MIRA artifact root containing
            ``temporal-instability/results.csv``.
        ignore_runner_slugs: Runner slugs to exclude before aggregation.

    Returns:
        Absolute paths to the generated artifacts.
    """
    csv_paths = find_metrics_csvs(metrics_folders)
    combined = concatenate_metrics(csv_paths)
    combined = _exclude_runner_slugs(
        combined,
        runner_column="runner",
        ignored=ignore_runner_slugs,
        data_description="MIRA metrics",
    )
    temporal_instability = _exclude_runner_slugs(
        read_temporal_instability_metrics(temporal_instability_mira_folder),
        runner_column="Runner",
        ignored=ignore_runner_slugs,
        data_description="temporal-instability results",
    )
    image_fidelity = _calculate_image_fidelity(temporal_instability)
    runner_gpu_quality = build_runner_gpu_quality_matrix(
        combined,
        image_fidelity,
    )
    generated_via_url, generated_via_label = _repository_provenance()

    resolved_output = _replace_results_output_dir(
        output_dir,
        input_csv_paths=csv_paths,
    )
    csv_path = resolved_output / "metrics_mira_combined.csv"
    matrix_csv_path = resolved_output / "runner_gpu_quality.csv"
    html_path = resolved_output / "mira_results.html"

    combined.to_csv(csv_path, index=False)
    runner_gpu_quality.to_csv(matrix_csv_path, index=False, float_format="%.3f")
    html_path.write_text(
        _render_html_report(
            runner_gpu_quality=runner_gpu_quality,
            generated_via_url=generated_via_url,
            generated_via_label=generated_via_label,
        ),
        encoding="utf-8",
    )
    return MiraResultsReport(
        csv_path=csv_path,
        matrix_csv_path=matrix_csv_path,
        html_path=html_path,
    )


def _replace_results_output_dir(
    output_dir: Path,
    *,
    input_csv_paths: tuple[Path, ...],
) -> Path:
    """Replace the report output directory without deleting report inputs."""
    resolved_output = output_dir.expanduser().resolve()
    resolved_workspace = Path.cwd().resolve()
    if resolved_workspace.is_relative_to(resolved_output):
        raise ValueError(
            "MIRA results output must not be the workspace or its parent: "
            f"{resolved_output}"
        )
    if any(csv_path.is_relative_to(resolved_output) for csv_path in input_csv_paths):
        raise ValueError(
            "MIRA results output must not contain an input metrics CSV: "
            f"{resolved_output}"
        )
    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError(
                f"MIRA results output exists and is not a directory: {resolved_output}"
            )
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True)
    return resolved_output


def _write_bar_chart(
    summary: pd.DataFrame,
    *,
    category_column: str = "Runner + GPU",
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
    alakazam_average_fps: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    chart_data = summary.copy()
    chart_data["_result_source"] = "Flashdreams"
    if alakazam_average_fps:
        alakazam_data = _read_alakazam_average_fps_rows()
        alakazam_data["_result_source"] = "Alakazam"
        chart_data = pd.concat([chart_data, alakazam_data], ignore_index=True)
        chart_data = _group_chart_rows_by_runner(chart_data)

    gpu_color_map: dict[str, str] = {}
    if category_column == "Runner + GPU":
        chart_data[category_column] = (
            chart_data["Runner"] + " | " + chart_data["GPU"].map(_chart_gpu_name)
        )
        flashdreams_gpus = chart_data.loc[
            chart_data["_result_source"] == "Flashdreams", "GPU"
        ]
        gpu_color_map = _gpu_color_map(flashdreams_gpus)
    figure_width = max(25.0, len(chart_data) * 1.25)
    axes = chart_data.plot.bar(
        x=category_column,
        y=value_column,
        color=color,
        figsize=(figure_width, 8),
        legend=False,
        rot=25,
    )
    if gpu_color_map:
        for bar, (_, row) in zip(
            axes.containers[0], chart_data.iterrows(), strict=True
        ):
            bar.set_facecolor(
                _ALAKAZAM_BAR_COLOR
                if row["_result_source"] == "Alakazam"
                else gpu_color_map[row["GPU"]]
            )
        legend_handles = [
            Patch(
                facecolor=gpu_color,
                label=f"{_chart_gpu_name(gpu_name)} - Flashdreams",
            )
            for gpu_name, gpu_color in gpu_color_map.items()
        ]
        if alakazam_average_fps:
            legend_handles.extend(
                Patch(
                    facecolor=_ALAKAZAM_BAR_COLOR,
                    label=f"{_chart_gpu_name(str(row['GPU']))} - Alakazam",
                )
                for _, row in alakazam_data.iterrows()
            )
        axes.legend(
            handles=legend_handles,
            title="GPU",
        )
    axes.set_title(title)
    axes.set_xlabel("")
    axes.set_ylabel(ylabel)
    axes.grid(axis="y", alpha=0.25)
    axes.bar_label(axes.containers[0], fmt="%.2f", padding=3)
    axes.figure.tight_layout()
    axes.figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(axes.figure)


def _read_alakazam_average_fps_rows() -> pd.DataFrame:
    """Read published average-FPS rows from the competition reference CSV."""
    reference = pd.read_csv(_COMPETITION_REFERENCE_PATH)
    required = {"runner", "gpu_name", "runtime_average_fps"}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(
            f"Competition reference CSV is missing required columns: {missing}"
        )
    rows = reference.loc[:, sorted(required)].rename(
        columns={
            "runner": "Runner",
            "gpu_name": "GPU",
            "runtime_average_fps": "Average FPS",
        }
    )
    rows["Average FPS"] = pd.to_numeric(rows["Average FPS"], errors="coerce")
    rows = rows.dropna(subset=["Runner", "GPU", "Average FPS"])
    if rows.empty:
        raise ValueError("Competition reference CSV contains no average-FPS rows.")
    return rows


def _gpu_color_map(gpu_names: pd.Series) -> dict[str, str]:
    """Assign stable green shades in GPU discovery order."""
    unique_names = list(dict.fromkeys(gpu_names.astype(str)))
    return {
        gpu_name: _GPU_BAR_COLORS[index % len(_GPU_BAR_COLORS)]
        for index, gpu_name in enumerate(unique_names)
    }


def _group_chart_rows_by_runner(chart_data: pd.DataFrame) -> pd.DataFrame:
    """Keep every result for the same runner adjacent in a chart."""
    return chart_data.sort_values("Runner", kind="stable", ignore_index=True)


def _calculate_image_fidelity(temporal_instability: pd.DataFrame) -> pd.DataFrame:
    """Express temporal instability relative to the minimum-value baseline."""
    image_fidelity = temporal_instability.loc[
        :, ["Runner", "Temporal Instability"]
    ].copy()
    instability = image_fidelity["Temporal Instability"]
    baseline = instability.min()
    image_fidelity["Image-Fidelity (%)"] = baseline / instability * 100.0
    image_fidelity.loc[
        instability == baseline,
        "Image-Fidelity (%)",
    ] = 100.0
    return image_fidelity.loc[:, ["Runner", "Image-Fidelity (%)"]].sort_values(
        ["Image-Fidelity (%)", "Runner"],
        ascending=[False, True],
        kind="stable",
        ignore_index=True,
    )


def build_runner_gpu_quality_matrix(
    metrics: pd.DataFrame,
    image_fidelity: pd.DataFrame,
) -> pd.DataFrame:
    """Build one runner row with dynamic GPU FPS columns and relative quality.

    Args:
        metrics: Concatenated raw MIRA metrics without synthetic reference rows.
        image_fidelity: Runner quality percentages derived from temporal instability.

    Returns:
        Matrix ordered as ``Runner-Slug``, GPU average-FPS columns, and ``Quality``.

    Raises:
        ValueError: Required columns are missing or no numeric FPS rows exist.
    """
    required_metrics = {"runner", "gpu_name", "runtime_average_fps"}
    missing_metrics = sorted(required_metrics.difference(metrics.columns))
    if missing_metrics:
        raise ValueError(f"MIRA metrics are missing matrix columns: {missing_metrics}")
    required_quality = {"Runner", "Image-Fidelity (%)"}
    missing_quality = sorted(required_quality.difference(image_fidelity.columns))
    if missing_quality:
        raise ValueError(
            f"Image-fidelity results are missing matrix columns: {missing_quality}"
        )

    fps_rows = metrics.loc[:, sorted(required_metrics)].copy()
    fps_rows["runtime_average_fps"] = pd.to_numeric(
        fps_rows["runtime_average_fps"],
        errors="coerce",
    )
    fps_rows = fps_rows.dropna(subset=["runner", "gpu_name", "runtime_average_fps"])
    if fps_rows.empty:
        raise ValueError("MIRA metrics contain no numeric average-FPS matrix rows.")
    fps_rows["gpu_name"] = fps_rows["gpu_name"].astype(str).map(_table_gpu_name)

    fps_matrix = (
        fps_rows.groupby(["runner", "gpu_name"], sort=True)["runtime_average_fps"]
        .mean()
        .unstack("gpu_name")
        .rename(columns=lambda gpu: f"{gpu} Avg. FPS")
        .rename_axis(columns=None)
        .reset_index()
        .rename(columns={"runner": "Runner-Slug"})
    )
    quality = image_fidelity.loc[:, ["Runner", "Image-Fidelity (%)"]].rename(
        columns={"Runner": "Runner-Slug", "Image-Fidelity (%)": "Quality"}
    )
    matrix = fps_matrix.merge(quality, on="Runner-Slug", how="outer", sort=True)
    gpu_columns = sorted(
        column for column in matrix.columns if column not in {"Runner-Slug", "Quality"}
    )
    return matrix.loc[:, ["Runner-Slug", *gpu_columns, "Quality"]].sort_values(
        ["Quality", "Runner-Slug"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )


def _table_gpu_name(gpu_name: str) -> str:
    """Return at most the first five words of a table GPU name."""
    return " ".join(gpu_name.split()[:5])


def _exclude_runner_slugs(
    data: pd.DataFrame,
    *,
    runner_column: str,
    ignored: tuple[str, ...],
    data_description: str,
) -> pd.DataFrame:
    """Remove ignored runner slugs and reject an empty result."""
    if not ignored:
        return data
    if runner_column not in data.columns:
        raise ValueError(
            f"{data_description} are missing runner column: {runner_column}"
        )
    filtered = data.loc[~data[runner_column].astype(str).isin(ignored)].copy()
    if filtered.empty:
        raise ValueError(f"Ignoring runner slugs removed all {data_description} rows.")
    return filtered.reset_index(drop=True)


def _repository_provenance() -> tuple[str, str]:
    """Return a web commit URL and label for the source repository."""
    repository_root = Path(__file__).resolve().parents[3]
    commit = _git_output(repository_root, "rev-parse", "HEAD")
    remote_url = _git_output(repository_root, "remote", "get-url", "upstream")
    if not remote_url:
        remote_url = _git_output(repository_root, "remote", "get-url", "origin")
    web_url = _repository_web_url(remote_url)
    if web_url and commit:
        return f"{web_url}/commit/{commit}", f"{web_url} @ {commit}"
    return "", "repository provenance unavailable"


def _git_output(repository_root: Path, *args: str) -> str:
    """Run a read-only Git query and return an empty string when unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _repository_web_url(remote_url: str) -> str:
    """Normalize common HTTPS and SSH Git remotes to a web repository URL."""
    normalized = remote_url.strip()
    if normalized.startswith("git@") and ":" in normalized:
        host, repository = normalized[4:].split(":", maxsplit=1)
        normalized = f"https://{host}/{repository}"
    return normalized.removesuffix(".git").rstrip("/")


def _chart_gpu_name(gpu_name: str) -> str:
    """Return at most the first four words of a GPU name."""
    return " ".join(gpu_name.split()[:4])


def _render_html_report(
    *,
    runner_gpu_quality: pd.DataFrame,
    generated_via_url: str,
    generated_via_label: str,
) -> str:
    matrix_table = _render_runner_gpu_quality_table(runner_gpu_quality)
    escaped_provenance_label = html.escape(generated_via_label)
    generated_via = (
        f'<a href="{html.escape(generated_via_url, quote=True)}">'
        f"{escaped_provenance_label}</a>"
        if generated_via_url
        else escaped_provenance_label
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MIRA Results Viewer</title>
  <style>
    body {{ background: #f4fbf6; color: #17351f;
            font: 15px system-ui, sans-serif; margin: 2rem; }}
    h1 {{ color: #14532d; }}
    .table-wrap {{ max-height: 36rem; overflow: auto; }}
    .metrics-table {{ border-collapse: collapse; min-width: 100%; white-space: nowrap; }}
    .metrics-table th {{ background: #166534; color: white;
                         position: sticky; top: 0; }}
    .metrics-table th, .metrics-table td {{ border: 1px solid #b7d6c0;
                                            padding: .45rem; }}
    .metrics-table tbody tr:nth-child(even) {{ background: #eaf6ed; }}
    .runner-gpu-quality td {{ font-variant-numeric: tabular-nums; }}
    .generated-via {{ color: #48624f; font-size: .9rem; }}
    a {{ color: #166534; }}
  </style>
</head>
<body>
  <h1>Runner performance and quality</h1>
  <div class="table-wrap">{matrix_table}</div>
  <p class="generated-via">Chart Generated via: {generated_via}</p>
</body>
</html>
"""


def _render_runner_gpu_quality_table(matrix: pd.DataFrame) -> str:
    """Render the runner matrix with threshold-colored average-FPS cells."""
    fps_columns = [column for column in matrix.columns if column.endswith(" Avg. FPS")]
    styler = matrix.style
    styler.hide(axis="index")
    for column in fps_columns:
        styler.format(
            "{:.2f}",
            subset=[column],
            na_rep="",
            escape="html",
        )
    styler.format(
        "{:.1f}%",
        subset=["Quality"],
        na_rep="",
        escape="html",
    )
    if fps_columns:
        styler.map(_average_fps_cell_style, subset=fps_columns)
    return styler.to_html(table_attributes='class="metrics-table runner-gpu-quality"')


def _average_fps_cell_style(value: Any) -> str:
    """Return a soft threshold color for one average-FPS table cell."""
    if pd.isna(value):
        return ""
    fps = float(value)
    if fps < 15.0:
        return "background-color: #f8d7da; color: #6b1d24;"
    if fps > 30.0:
        return "background-color: #d9f2df; color: #174d28;"
    return "background-color: #fff3cd; color: #5f4b00;"


__all__ = [
    "MiraResultsReport",
    "MiraResultsViewer",
    "MiraResultsViewerConfig",
    "build_results_report",
    "build_runner_gpu_quality_matrix",
    "concatenate_metrics",
    "find_metrics_csvs",
    "read_temporal_instability_metrics",
    "summarize_runner_gpu_metrics",
]
