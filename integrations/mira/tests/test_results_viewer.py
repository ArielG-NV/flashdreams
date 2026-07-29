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

"""CPU tests for the MIRA pandas results viewer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import tyro
from mira_integration.results_viewer import (
    MiraResultsViewer,
    MiraResultsViewerConfig,
    _calculate_image_fidelity,
    _chart_gpu_name,
    _group_chart_rows_by_runner,
    _pareto_curve_points,
    _render_runner_gpu_quality_table,
    build_results_report,
    build_runner_gpu_quality_matrix,
    find_metrics_csvs,
    read_temporal_instability_metrics,
    summarize_runner_gpu_metrics,
)

pytestmark = pytest.mark.ci_cpu


def _write_metrics(
    folder: Path,
    slug: str,
    *,
    gpu: str,
    fps: float,
    one_percent_lows_fps: float,
    model_vram_gib: float,
) -> Path:
    output = folder / slug / "metrics_mira.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "runner": slug,
                "gpu_name": gpu,
                "runtime_average_fps": fps,
                "runtime_1_percent_lows_fps": one_percent_lows_fps,
                "model_load_vram_gib": model_vram_gib,
            }
        ]
    ).to_csv(output, index=False)
    return output.resolve()


def _write_temporal_instability_metrics(
    mira_folder: Path,
    rows: list[dict[str, str | float]],
) -> Path:
    output = mira_folder / "temporal-instability" / "results.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.resolve()


def test_find_metrics_csvs_searches_each_direct_slug(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    csv_b = _write_metrics(
        first, "b", gpu="GPU 1", fps=20, one_percent_lows_fps=18, model_vram_gib=8
    )
    csv_a = _write_metrics(
        first, "a", gpu="GPU 1", fps=10, one_percent_lows_fps=8, model_vram_gib=7
    )
    csv_c = _write_metrics(
        second, "c", gpu="GPU 2", fps=30, one_percent_lows_fps=27, model_vram_gib=9
    )
    _write_metrics(
        first / "nested",
        "ignored",
        gpu="GPU 3",
        fps=40,
        one_percent_lows_fps=35,
        model_vram_gib=6,
    )

    assert find_metrics_csvs((first, second)) == (csv_a, csv_b, csv_c)


def test_viewer_config_parses_metrics_folders_positionally() -> None:
    config = tyro.cli(
        MiraResultsViewerConfig,
        args=[
            "metrics_folder_1",
            "metrics_folder_2",
            "--temporal-instability-mira-folder",
            "mira_folder",
        ],
        default=MiraResultsViewerConfig(runner_name="mira-results-viewer"),
    )

    assert config.metrics_folders == (
        Path("metrics_folder_1"),
        Path("metrics_folder_2"),
    )


def test_viewer_config_parses_temporal_instability_mira_folder() -> None:
    config = tyro.cli(
        MiraResultsViewerConfig,
        args=[
            "metrics_folder",
            "--temporal-instability-mira-folder",
            "mira_folder",
        ],
        default=MiraResultsViewerConfig(runner_name="mira-results-viewer"),
    )

    assert config.temporal_instability_mira_folder == Path("mira_folder")


def test_viewer_config_requires_temporal_instability_mira_folder() -> None:
    with pytest.raises(SystemExit):
        tyro.cli(
            MiraResultsViewerConfig,
            args=["metrics_folder"],
            default=MiraResultsViewerConfig(runner_name="mira-results-viewer"),
        )


def test_viewer_config_parses_ignored_runner_slugs() -> None:
    config = tyro.cli(
        MiraResultsViewerConfig,
        args=[
            "metrics_folder",
            "--temporal-instability-mira-folder",
            "mira_folder",
            "--ignore-runner-slug",
            "mira-a",
            "mira-b",
        ],
        default=MiraResultsViewerConfig(runner_name="mira-results-viewer"),
    )

    assert config.ignore_runner_slug == ("mira-a", "mira-b")


def test_summarize_runner_gpu_metrics_averages_repeated_configs() -> None:
    metrics = pd.DataFrame(
        [
            {
                "runner": "mira-a",
                "gpu_name": "GPU 1",
                "runtime_average_fps": "40",
                "runtime_1_percent_lows_fps": "35",
                "model_load_vram_gib": "7",
            },
            {
                "runner": "mira-a",
                "gpu_name": "GPU 1",
                "runtime_average_fps": "60",
                "runtime_1_percent_lows_fps": "50",
                "model_load_vram_gib": "9",
            },
            {
                "runner": "mira-a",
                "gpu_name": "GPU 2",
                "runtime_average_fps": "70",
                "runtime_1_percent_lows_fps": "60",
                "model_load_vram_gib": "10",
            },
        ]
    )

    summary = summarize_runner_gpu_metrics(metrics)

    assert summary["Runner + GPU"].tolist() == [
        "mira-a | GPU 1",
        "mira-a | GPU 2",
    ]
    assert summary["Average FPS"].tolist() == [50.0, 70.0]
    assert summary["Average FPS 1% Lows"].tolist() == [42.5, 60.0]
    assert summary["Model VRAM Footprint (GiB)"].tolist() == [8.0, 10.0]


@pytest.mark.parametrize(
    ("gpu_name", "expected"),
    [
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 6000"),
        ("NVIDIA H100", "NVIDIA H100"),
        ("  NVIDIA   A100  80GB   PCIe  ", "NVIDIA A100 80GB PCIe"),
    ],
)
def test_chart_gpu_name_is_capped_at_four_words(
    gpu_name: str,
    expected: str,
) -> None:
    assert _chart_gpu_name(gpu_name) == expected


def test_chart_rows_are_grouped_by_runner() -> None:
    chart_data = pd.DataFrame(
        [
            {"Runner": "mira-a", "GPU": "B200"},
            {"Runner": "mira-mini-1-player-1b-8-step", "GPU": "B200"},
            {"Runner": "mira-z", "GPU": "B200"},
            {"Runner": "mira-mini-1-player-1b-8-step", "GPU": "B200"},
            {"Runner": "mira-mini-1-player-1b-8-step", "GPU": "M1 Pro"},
        ]
    )

    grouped = _group_chart_rows_by_runner(chart_data)

    assert grouped["Runner"].tolist() == [
        "mira-a",
        "mira-mini-1-player-1b-8-step",
        "mira-mini-1-player-1b-8-step",
        "mira-mini-1-player-1b-8-step",
        "mira-z",
    ]
    assert grouped["GPU"].tolist()[1:4] == ["B200", "B200", "M1 Pro"]


def test_read_temporal_instability_metrics_averages_repeated_runners(
    tmp_path: Path,
) -> None:
    _write_temporal_instability_metrics(
        tmp_path,
        [
            {"runner": "mira-b", "temporal_instability_metric": "4.5"},
            {"runner": "mira-a", "temporal_instability_metric": "2.0"},
            {"runner": "mira-a", "temporal_instability_metric": "4.0"},
        ],
    )

    summary = read_temporal_instability_metrics(tmp_path)

    assert summary["Runner"].tolist() == ["mira-a", "mira-b"]
    assert summary["Temporal Instability"].tolist() == [3.0, 4.5]


def test_image_fidelity_uses_minimum_as_baseline_and_sorts_descending() -> None:
    temporal_instability = pd.DataFrame(
        [
            {"Runner": "mira-c", "Temporal Instability": 4.0},
            {"Runner": "mira-a", "Temporal Instability": 2.0},
            {"Runner": "mira-b", "Temporal Instability": 2.5},
        ]
    )

    image_fidelity = _calculate_image_fidelity(temporal_instability)

    assert image_fidelity["Runner"].tolist() == ["mira-a", "mira-b", "mira-c"]
    assert image_fidelity["Image-Fidelity (%)"].tolist() == [100.0, 80.0, 50.0]
    assert image_fidelity["Temporal Instability"].tolist() == [2.0, 2.5, 4.0]


def test_runner_gpu_quality_matrix_pivots_real_metrics_only() -> None:
    metrics = pd.DataFrame(
        [
            {
                "runner": "mira-a",
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "runtime_average_fps": 20.0,
            },
            {
                "runner": "mira-a",
                "gpu_name": "GPU 1",
                "runtime_average_fps": 10.0,
            },
            {
                "runner": "mira-a",
                "gpu_name": "GPU 1",
                "runtime_average_fps": 14.0,
            },
            {
                "runner": "mira-b",
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "runtime_average_fps": 40.0,
            },
        ]
    )
    image_fidelity = pd.DataFrame(
        [
            {
                "Runner": "mira-a",
                "Image-Fidelity (%)": 50.0,
                "Temporal Instability": 4.0,
            },
            {
                "Runner": "mira-b",
                "Image-Fidelity (%)": 100.0,
                "Temporal Instability": 2.0,
            },
        ]
    )

    matrix = build_runner_gpu_quality_matrix(metrics, image_fidelity)

    assert matrix.columns.tolist() == [
        "Runner-Slug",
        "GPU 1 Avg. FPS",
        "NVIDIA RTX PRO 6000 Avg. FPS",
        "Quality",
        "Temporal Stability",
    ]
    assert matrix["Runner-Slug"].tolist() == ["mira-b", "mira-a"]
    assert pd.isna(matrix.loc[0, "GPU 1 Avg. FPS"])
    assert matrix.loc[0, "NVIDIA RTX PRO 6000 Avg. FPS"] == 40.0
    assert matrix.loc[1, "GPU 1 Avg. FPS"] == 12.0
    assert matrix.loc[1, "NVIDIA RTX PRO 6000 Avg. FPS"] == 20.0
    assert matrix["Quality"].tolist() == [100.0, 50.0]
    assert matrix["Temporal Stability"].tolist() == [2.0, 4.0]


def test_runner_gpu_quality_table_colors_fps_thresholds() -> None:
    matrix = pd.DataFrame(
        [
            {
                "Runner-Slug": "mira-a",
                "GPU 1 Avg. FPS": 14.99,
                "GPU 2 Avg. FPS": 15.0,
                "GPU 3 Avg. FPS": 30.01,
                "Quality": 100.0,
                "Temporal Stability": 2.34567,
            }
        ]
    )

    rendered = _render_runner_gpu_quality_table(matrix)

    assert "#f8d7da" in rendered
    assert "#fff3cd" in rendered
    assert "#d9f2df" in rendered
    assert "Quality<sup>1</sup>" in rendered
    assert "100.0% (2.34567)" in rendered
    assert ">Temporal Stability<" not in rendered


def test_pareto_curve_points_excludes_dominated_candidates() -> None:
    matrix = pd.DataFrame(
        [
            {
                "Runner-Slug": "quality",
                "GPU 1 Avg. FPS": 10.0,
                "Quality": 100.0,
                "Temporal Stability": 2.0,
            },
            {
                "Runner-Slug": "balanced",
                "GPU 1 Avg. FPS": 20.0,
                "Quality": 90.0,
                "Temporal Stability": 2.2,
            },
            {
                "Runner-Slug": "dominated",
                "GPU 1 Avg. FPS": 15.0,
                "Quality": 80.0,
                "Temporal Stability": 2.5,
            },
            {
                "Runner-Slug": "speed",
                "GPU 1 Avg. FPS": 30.0,
                "Quality": 70.0,
                "Temporal Stability": 2.9,
            },
        ]
    )

    frontier = _pareto_curve_points(matrix, gpu="GPU 1")

    assert frontier["Runner-Slug"].tolist() == ["quality", "balanced", "speed"]


def test_pareto_curve_points_are_calculated_per_gpu() -> None:
    matrix = pd.DataFrame(
        [
            {
                "Runner-Slug": "gpu-1-frontier",
                "GPU 1 Avg. FPS": 10.0,
                "GPU 2 Avg. FPS": 40.0,
                "Quality": 100.0,
                "Temporal Stability": 2.0,
            },
            {
                "Runner-Slug": "gpu-2-frontier",
                "GPU 1 Avg. FPS": 20.0,
                "GPU 2 Avg. FPS": 30.0,
                "Quality": 90.0,
                "Temporal Stability": 2.2,
            },
        ]
    )

    gpu_1_frontier = _pareto_curve_points(matrix, gpu="GPU 1")
    gpu_2_frontier = _pareto_curve_points(matrix, gpu="GPU 2")

    assert gpu_1_frontier["Runner-Slug"].tolist() == [
        "gpu-1-frontier",
        "gpu-2-frontier",
    ]
    assert gpu_2_frontier["Runner-Slug"].tolist() == ["gpu-1-frontier"]


def test_build_results_report_writes_csv_table_and_pareto_svg(
    tmp_path: Path,
) -> None:
    metrics_folder = tmp_path / "metrics"
    first = _write_metrics(
        metrics_folder,
        "mira-a",
        gpu="NVIDIA Test GPU",
        fps=50,
        one_percent_lows_fps=45,
        model_vram_gib=9,
    )
    _write_metrics(
        metrics_folder,
        "mira-b",
        gpu="NVIDIA Test GPU",
        fps=70,
        one_percent_lows_fps=60,
        model_vram_gib=12,
    )
    _write_temporal_instability_metrics(
        tmp_path / "mira",
        [
            {"runner": "mira-a", "temporal_instability_metric": 2.5},
            {"runner": "mira-b", "temporal_instability_metric": 3.5},
        ],
    )

    report = build_results_report(
        (metrics_folder,),
        output_dir=tmp_path / "report",
        temporal_instability_mira_folder=tmp_path / "mira",
    )

    assert report.csv_path.is_file()
    assert report.matrix_csv_path.is_file()
    assert len(report.pareto_curve_paths) == 1
    assert report.pareto_curve_paths[0].is_file()
    assert (
        report.pareto_curve_paths[0].name
        == "pareto_curve_mira_mini_nvidia-test-gpu.svg"
    )
    assert report.html_path.is_file()
    assert list(report.html_path.parent.glob("*.svg")) == [report.pareto_curve_paths[0]]
    combined = pd.read_csv(report.csv_path)
    matrix = pd.read_csv(report.matrix_csv_path)
    assert len(combined) == 2
    assert combined.loc[0, "source_csv"] == str(first)
    assert matrix.columns.tolist() == [
        "Runner-Slug",
        "NVIDIA Test GPU Avg. FPS",
        "Quality",
        "Temporal Stability",
    ]
    assert matrix["Runner-Slug"].tolist() == ["mira-a", "mira-b"]
    assert matrix["Quality"].tolist() == pytest.approx([100.0, 71.429], abs=0.001)
    rendered = report.html_path.read_text(encoding="utf-8")
    assert "Runner performance and quality" in rendered
    assert "/commit/" in rendered
    assert rendered.count("<table") == 1
    assert "<img" not in rendered
    assert "pareto_curve_mira_mini" not in rendered
    assert "Quality<sup>1</sup>" in rendered
    assert "<sup>1</sup> Quality: defined as temporal stability." in rendered
    assert "100.0% (2.5)" in rendered
    assert "Runner + GPU averages" not in rendered
    assert "Concatenated metrics" not in rendered
    pareto_svg = report.pareto_curve_paths[0].read_text(encoding="utf-8")
    assert "NVIDIA Test GPU" in pareto_svg
    assert "Quality¹ (%)" in pareto_svg
    assert "¹ Quality: defined as temporal stability." in pareto_svg


def test_build_results_report_ignores_runner_before_quality_baseline(
    tmp_path: Path,
) -> None:
    metrics_folder = tmp_path / "metrics"
    _write_metrics(
        metrics_folder,
        "mira-a",
        gpu="GPU 1",
        fps=10,
        one_percent_lows_fps=8,
        model_vram_gib=7,
    )
    _write_metrics(
        metrics_folder,
        "mira-b",
        gpu="GPU 1",
        fps=20,
        one_percent_lows_fps=18,
        model_vram_gib=7,
    )
    _write_temporal_instability_metrics(
        tmp_path / "mira",
        [
            {"runner": "mira-a", "temporal_instability_metric": 2.0},
            {"runner": "mira-b", "temporal_instability_metric": 4.0},
        ],
    )

    report = build_results_report(
        (metrics_folder,),
        output_dir=tmp_path / "report",
        temporal_instability_mira_folder=tmp_path / "mira",
        ignore_runner_slugs=("mira-a",),
    )

    matrix = pd.read_csv(report.matrix_csv_path)
    assert matrix["Runner-Slug"].tolist() == ["mira-b"]
    assert matrix["Quality"].tolist() == [100.0]


def test_runner_gpu_quality_csv_excludes_alakazam_reference_rows(
    tmp_path: Path,
) -> None:
    runner = "mira-mini-1-player-1b-8-step"
    metrics_folder = tmp_path / "metrics"
    _write_metrics(
        metrics_folder,
        runner,
        gpu="Measured GPU",
        fps=12,
        one_percent_lows_fps=10,
        model_vram_gib=7,
    )
    _write_temporal_instability_metrics(
        tmp_path / "mira",
        [{"runner": runner, "temporal_instability_metric": 2.0}],
    )

    report = build_results_report(
        (metrics_folder,),
        output_dir=tmp_path / "report",
        temporal_instability_mira_folder=tmp_path / "mira",
    )

    matrix = pd.read_csv(report.matrix_csv_path)
    assert matrix.columns.tolist() == [
        "Runner-Slug",
        "Measured GPU Avg. FPS",
        "Quality",
        "Temporal Stability",
    ]
    assert matrix.loc[0, "Measured GPU Avg. FPS"] == 12.0


def test_viewer_prints_and_opens_local_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics_folder = tmp_path / "metrics"
    _write_metrics(
        metrics_folder,
        "mira-a",
        gpu="NVIDIA Test GPU",
        fps=50,
        one_percent_lows_fps=45,
        model_vram_gib=9,
    )
    _write_temporal_instability_metrics(
        tmp_path / "mira",
        [{"runner": "mira-a", "temporal_instability_metric": 2.0}],
    )
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda uri: opened.append(uri) or True)
    config = MiraResultsViewerConfig(
        runner_name="mira-results-viewer",
        metrics_folders=(metrics_folder,),
        temporal_instability_mira_folder=tmp_path / "mira",
        output_dir=tmp_path / "report",
    )
    viewer = config.setup()
    assert isinstance(viewer, MiraResultsViewer)

    viewer.run()

    output = capsys.readouterr().out
    assert "Combined CSV:" in output
    assert "Runner/GPU/quality CSV:" in output
    assert "Rendered in your default web browser from:" in output
    assert opened == [
        (tmp_path / "report" / "general_mira_performance_results.html")
        .resolve()
        .as_uri()
    ]


def test_find_metrics_csvs_rejects_missing_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No MIRA metrics files"):
        find_metrics_csvs((tmp_path,))
