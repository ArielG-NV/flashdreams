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

"""CPU tests for MIRA comparison chart generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import tyro
from mira_integration.comparison_chart_generator import (
    MiraComparisonChartGeneratorConfig,
    _comparison_figure_height,
    _legend_gpu_name,
    build_comparison_chart,
    collect_comparison_bars,
    read_competition_reference,
)

pytestmark = pytest.mark.ci_cpu

_DIRECT_RUNNER = "mira-mini-1-player-1b-8-step"


def _write_metrics(
    mira_folder: Path,
    runner_slug: str,
    *,
    gpu_name: str = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    metric: str = "runtime_average_fps",
    values: tuple[float, ...] = (10.0,),
) -> Path:
    output = mira_folder / runner_slug / "metrics_mira.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"runner": runner_slug, "gpu_name": gpu_name, metric: value}
            for value in values
        ]
    ).to_csv(output, index=False)
    return output


def test_packaged_competition_reference_has_sources() -> None:
    reference = read_competition_reference()

    assert reference.columns.tolist() == [
        "runner",
        "gpu_name",
        "runtime_average_fps",
        "reference",
    ]
    assert reference["reference"].str.startswith("https://").all()
    rtx_reference = reference.loc[
        reference["gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell", "reference"
    ].item()
    assert rtx_reference == "https://github.com/ArielG-NV/alakazam-mira-mini"
    assert set(
        reference.loc[
            reference["gpu_name"] != "NVIDIA RTX PRO 6000 Blackwell",
            "reference",
        ]
    ) == {"https://huggingface.co/alakazamworld/mira-mini"}


def test_config_parses_repeated_other_runner_flags() -> None:
    config = tyro.cli(
        MiraComparisonChartGeneratorConfig,
        args=[
            "artifacts/mira",
            "--metric-to-compare",
            "runtime_average_fps",
            "--runner-slug-direct-compare",
            _DIRECT_RUNNER,
            "--flashdreams-gpu-to-compare-with",
            "RTX PRO 6000",
            "--competitor-gpu-to-compare-with",
            "B200",
            "--custom-y-axis",
            "Average FPS",
            "--custom-title",
            "MIRA throughput comparison",
            "--flashdreams-gpu-other-runner",
            "other-a",
            "--flashdreams-gpu-other-runner",
            "other-b",
        ],
        default=MiraComparisonChartGeneratorConfig(
            runner_name="mira-comparison-chart-generator"
        ),
    )

    assert config.flashdreams_gpu_other_runner == ("other-a", "other-b")
    assert config.custom_y_axis == "Average FPS"
    assert config.custom_title == "MIRA throughput comparison"


def test_config_requires_custom_y_axis_and_title() -> None:
    with pytest.raises(SystemExit):
        tyro.cli(
            MiraComparisonChartGeneratorConfig,
            args=[
                "artifacts/mira",
                "--metric-to-compare",
                "runtime_average_fps",
                "--runner-slug-direct-compare",
                _DIRECT_RUNNER,
                "--flashdreams-gpu-to-compare-with",
                "RTX PRO 6000",
                "--competitor-gpu-to-compare-with",
                "B200",
            ],
            default=MiraComparisonChartGeneratorConfig(
                runner_name="mira-comparison-chart-generator"
            ),
        )


def test_collects_competitor_first_and_averages_measured_rows(
    tmp_path: Path,
) -> None:
    _write_metrics(tmp_path, _DIRECT_RUNNER, values=(10.0, 14.0))
    _write_metrics(tmp_path, "other-runner", values=(20.0,))

    bars = collect_comparison_bars(
        tmp_path,
        metric_to_compare="runtime_average_fps",
        runner_slug_direct_compare=_DIRECT_RUNNER,
        flashdreams_gpu_to_compare_with=r"RTX PRO 6000",
        competitor_gpu_to_compare_with=r"B\d+",
        flashdreams_gpu_other_runners=("other-runner",),
    )

    assert [bar.runner_slug for bar in bars] == [
        _DIRECT_RUNNER,
        _DIRECT_RUNNER,
        "other-runner",
    ]
    assert [bar.source for bar in bars] == [
        "Competitor",
        "FlashDreams",
        "FlashDreams",
    ]
    assert [bar.value for bar in bars] == [25.7, 12.0, 20.0]
    assert bars[0].reference == "https://huggingface.co/alakazamworld/mira-mini"


def test_competitor_gpu_regex_rejects_multiple_matches(tmp_path: Path) -> None:
    _write_metrics(tmp_path, _DIRECT_RUNNER)

    with pytest.raises(ValueError, match="matched multiple GPUs"):
        collect_comparison_bars(
            tmp_path,
            metric_to_compare="runtime_average_fps",
            runner_slug_direct_compare=_DIRECT_RUNNER,
            flashdreams_gpu_to_compare_with="RTX PRO 6000",
            competitor_gpu_to_compare_with=r"B200|M1 Pro",
        )


def test_competitor_gpu_regex_rejects_invalid_expression(tmp_path: Path) -> None:
    _write_metrics(tmp_path, _DIRECT_RUNNER)

    with pytest.raises(
        ValueError,
        match="Invalid --competitor-gpu-to-compare-with regular expression",
    ):
        collect_comparison_bars(
            tmp_path,
            metric_to_compare="runtime_average_fps",
            runner_slug_direct_compare=_DIRECT_RUNNER,
            flashdreams_gpu_to_compare_with="RTX PRO 6000",
            competitor_gpu_to_compare_with="[",
        )


def test_build_comparison_chart_writes_ordered_colored_svg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    legend_options: dict[str, object] = {}
    citation_geometry: dict[str, object] = {}
    original_legend = Axes.legend
    original_text = Axes.text

    def capture_legend_options(
        axes: Axes,
        *args: object,
        **kwargs: object,
    ) -> object:
        legend_options.update(kwargs)
        return original_legend(axes, *args, **kwargs)

    def capture_citation_geometry(
        axes: Axes,
        x: float,
        y: float,
        text: str,
        fontdict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> object:
        if text.startswith("¹ "):
            citation_geometry.update(x=x, y=y, **kwargs)
        return original_text(axes, x, y, text, fontdict=fontdict, **kwargs)

    monkeypatch.setattr(Axes, "legend", capture_legend_options)
    monkeypatch.setattr(Axes, "text", capture_citation_geometry)
    _write_metrics(tmp_path, _DIRECT_RUNNER)
    _write_metrics(tmp_path, "other-runner", values=(20.0,))

    output = build_comparison_chart(
        tmp_path,
        metric_to_compare="runtime_average_fps",
        runner_slug_direct_compare=_DIRECT_RUNNER,
        flashdreams_gpu_to_compare_with=r"RTX PRO 6000",
        competitor_gpu_to_compare_with="B200",
        custom_y_axis="Average FPS",
        custom_title="MIRA throughput comparison",
        flashdreams_gpu_other_runners=("other-runner",),
        output_path=tmp_path / "chart.svg",
    )

    rendered = output.read_text(encoding="utf-8")
    assert output == (tmp_path / "chart.svg").resolve()
    assert "#d62728" in rendered
    assert "#2e8b57" in rendered
    assert "B200 - Competitor¹" in rendered
    assert "B200 - competitor" not in rendered
    assert "NVIDIA RTX PRO 6000 Blackwell - FlashDreams" in rendered
    assert "Workstation Edition - FlashDreams" not in rendered
    assert "Average FPS" in rendered
    assert "MIRA throughput comparison" in rendered
    assert "¹ https://huggingface.co/alakazamworld/mira-mini" in rendered
    assert 'xlink:href="https://huggingface.co/alakazamworld/mira-mini"' in rendered
    assert rendered.index(_DIRECT_RUNNER) < rendered.index("other-runner")
    assert legend_options["loc"] == "center"
    assert "bbox_to_anchor" not in legend_options
    assert legend_options["ncols"] == 1
    assert citation_geometry["x"] == 0.5
    assert citation_geometry["y"] == 0.5
    assert citation_geometry["ha"] == "center"
    assert citation_geometry["va"] == "center"


def test_comparison_figure_height_grows_with_runner_label_length() -> None:
    short_height = _comparison_figure_height(("short-runner",))
    long_height = _comparison_figure_height(("runner-" + "x" * 100,))

    assert short_height == 9.5
    assert long_height > short_height


@pytest.mark.parametrize(
    ("gpu_name", "expected"),
    [
        (
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell",
        ),
        ("B200", "B200"),
        ("  NVIDIA   H100   NVL  ", "NVIDIA H100 NVL"),
    ],
)
def test_legend_gpu_name_is_capped_at_five_words(
    gpu_name: str,
    expected: str,
) -> None:
    assert _legend_gpu_name(gpu_name) == expected


@pytest.mark.parametrize(
    ("custom_y_axis", "custom_title", "message"),
    [
        (" ", "Chart title", "--custom-y-axis must not be blank"),
        ("Average FPS", " ", "--custom-title must not be blank"),
    ],
)
def test_build_comparison_chart_rejects_blank_presentation_text(
    tmp_path: Path,
    custom_y_axis: str,
    custom_title: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_comparison_chart(
            tmp_path,
            metric_to_compare="runtime_average_fps",
            runner_slug_direct_compare=_DIRECT_RUNNER,
            flashdreams_gpu_to_compare_with="RTX PRO 6000",
            competitor_gpu_to_compare_with="B200",
            custom_y_axis=custom_y_axis,
            custom_title=custom_title,
            output_path=tmp_path / "chart.svg",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"metric_to_compare": "missing_metric"},
            "missing from competition reference CSV",
        ),
        (
            {"competitor_gpu_to_compare_with": "Missing GPU"},
            "has no GPU matching",
        ),
        (
            {"flashdreams_gpu_to_compare_with": "H100"},
            "has no GPU matching",
        ),
        (
            {"runner_slug_direct_compare": "missing-runner"},
            "has no row",
        ),
    ],
)
def test_missing_comparison_data_raises_clear_error(
    tmp_path: Path,
    kwargs: dict[str, str],
    message: str,
) -> None:
    _write_metrics(tmp_path, _DIRECT_RUNNER)
    with pytest.raises(ValueError, match=message):
        collect_comparison_bars(
            tmp_path,
            metric_to_compare=kwargs.get("metric_to_compare", "runtime_average_fps"),
            runner_slug_direct_compare=kwargs.get(
                "runner_slug_direct_compare", _DIRECT_RUNNER
            ),
            flashdreams_gpu_to_compare_with=kwargs.get(
                "flashdreams_gpu_to_compare_with", "RTX PRO 6000"
            ),
            competitor_gpu_to_compare_with=kwargs.get(
                "competitor_gpu_to_compare_with", "B200"
            ),
        )


def test_missing_other_runner_raises_clear_error(tmp_path: Path) -> None:
    _write_metrics(tmp_path, _DIRECT_RUNNER)

    with pytest.raises(ValueError, match="other-runner.*has no metrics file"):
        collect_comparison_bars(
            tmp_path,
            metric_to_compare="runtime_average_fps",
            runner_slug_direct_compare=_DIRECT_RUNNER,
            flashdreams_gpu_to_compare_with="RTX PRO 6000",
            competitor_gpu_to_compare_with="B200",
            flashdreams_gpu_other_runners=("other-runner",),
        )
