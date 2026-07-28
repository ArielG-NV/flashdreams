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
    _chart_gpu_name,
    _group_chart_rows_by_runner,
    build_results_report,
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
        args=["metrics_folder_1", "metrics_folder_2"],
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
    assert summary["1% Lows FPS"].tolist() == [42.5, 60.0]
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
            {"Runner": "mira-mini-1p-1b-high", "GPU": "B200"},
            {"Runner": "mira-z", "GPU": "B200"},
            {"Runner": "mira-mini-1p-1b-high", "GPU": "B200"},
            {"Runner": "mira-mini-1p-1b-high", "GPU": "M1 Pro"},
        ]
    )

    grouped = _group_chart_rows_by_runner(chart_data)

    assert grouped["Runner"].tolist() == [
        "mira-a",
        "mira-mini-1p-1b-high",
        "mira-mini-1p-1b-high",
        "mira-mini-1p-1b-high",
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


def test_build_results_report_writes_csv_html_and_charts(tmp_path: Path) -> None:
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
    assert report.html_path.is_file()
    assert report.fps_chart_path.is_file()
    assert report.one_percent_lows_fps_chart_path.is_file()
    assert report.model_vram_chart_path.is_file()
    assert report.temporal_instability_chart_path is not None
    assert report.temporal_instability_chart_path.is_file()
    combined = pd.read_csv(report.csv_path)
    assert len(combined) == 2
    assert combined.loc[0, "source_csv"] == str(first)
    rendered = report.html_path.read_text(encoding="utf-8")
    assert "Pandas generated this local report" in rendered
    assert "Average FPS" in rendered
    assert report.fps_chart_path.name in rendered
    assert report.one_percent_lows_fps_chart_path.name in rendered
    assert report.model_vram_chart_path.name in rendered
    assert report.temporal_instability_chart_path.name in rendered
    assert "1% Lows FPS" in rendered
    assert "VRAM Footprint Of Model Config" in rendered
    assert "Temporal Instability" in rendered


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
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda uri: opened.append(uri) or True)
    config = MiraResultsViewerConfig(
        runner_name="mira-results-viewer",
        metrics_folders=(metrics_folder,),
        output_dir=tmp_path / "report",
    )
    viewer = config.setup()
    assert isinstance(viewer, MiraResultsViewer)

    viewer.run()

    output = capsys.readouterr().out
    assert "Combined CSV:" in output
    assert "Rendered in your default web browser from:" in output
    assert opened == [(tmp_path / "report" / "mira_results.html").resolve().as_uri()]


def test_find_metrics_csvs_rejects_missing_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No MIRA metrics files"):
        find_metrics_csvs((tmp_path,))
