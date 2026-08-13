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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
    T2VApplicationSession,
)

from flashdreams.demo import (
    CanonicalInputs,
    Mp4OutputSink,
    NullInputHandler,
    NullOutputSink,
    OutputDecision,
    ProvidedIOFactory,
)
from flashdreams.demo import application as application_module
from flashdreams.infra.results import StepResult
from flashdreams.runtime import CanonicalInputSchema, CanonicalModality

pytestmark = pytest.mark.ci_cpu


class _FakeDecoder:
    spatial_compression_ratio = 8


class _FakePipeline:
    def __init__(self) -> None:
        self.decoder = _FakeDecoder()
        self.device: str | None = None
        self.cache_kwargs: dict[str, Any] | None = None
        self.generated: list[int] = []
        self.finalized: list[int] = []
        self.closed = False

    def to(self, device: str) -> "_FakePipeline":
        self.device = device
        return self

    def eval(self) -> "_FakePipeline":
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.cache_kwargs = kwargs
        return object()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        del autoregressive_index
        return 2

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
    ) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        return torch.full((2, 3, 4, 5), float(autoregressive_index))

    def finalize(
        self,
        *,
        autoregressive_index: int,
        cache: object,
    ) -> dict[str, float]:
        del cache
        self.finalized.append(autoregressive_index)
        return {"model_step_s": 0.25}

    def close(self) -> None:
        self.closed = True


class _FakePipelineConfig:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> _FakePipeline:
        return self.pipeline


class _StoppingSink(NullOutputSink):
    def write(self, result: StepResult) -> OutputDecision:
        super().write(result)
        return OutputDecision(should_stop=True)


def _application(pipeline: _FakePipeline) -> T2VApplication:
    return T2VApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=_FakePipelineConfig(pipeline),
            total_blocks=4,
            pixel_height=480,
            pixel_width=832,
        )
    )


def _initialize_application(
    application: T2VApplication,
    pipeline: _FakePipeline,
    output_sink: NullOutputSink,
    *,
    total_blocks: int = 2,
) -> tuple[NullInputHandler, T2VApplicationSession]:
    input_handler = NullInputHandler()
    application.init(
        [
            "--prompt",
            "A waterfall",
            "--total-blocks",
            str(total_blocks),
            "--device",
            "cpu",
        ]
    )
    session = application.createSession()
    assert isinstance(session, T2VApplicationSession)
    session.init()
    session_info = session.session_info()
    input_handler.open(session_info)
    output_sink.open(session_info)
    output_sink.begin_generation(0)
    assert pipeline.device == "cpu"
    return input_handler, session


def test_prompt_is_required() -> None:
    application = _application(_FakePipeline())
    with pytest.raises(ValueError, match="--prompt is required"):
        application.init([])


def test_application_session_emits_canonical_video_results() -> None:
    pipeline = _FakePipeline()
    output_sink = NullOutputSink(store_results=True, store_outputs=True)
    application = _application(pipeline)
    input_handler, session = _initialize_application(
        application,
        pipeline,
        output_sink,
    )

    while session.step(CanonicalInputs(), output_sink):
        pass
    session.close()
    output_sink.close()
    input_handler.close()

    assert pipeline.cache_kwargs == {
        "text": ["A waterfall"],
        "image": None,
        "height": 60,
        "width": 104,
    }
    assert pipeline.generated == [0, 1]
    assert pipeline.finalized == [0, 1]
    assert [tuple(output.shape) for output in output_sink.outputs] == [
        (2, 3, 4, 5),
        (2, 3, 4, 5),
    ]
    assert [record["layout"] for record in output_sink.results] == ["tchw", "tchw"]
    assert output_sink.session_info is not None
    assert output_sink.session_info.frames_per_second == 16
    assert output_sink.session_info.video_width == 832
    assert output_sink.session_info.video_height == 480
    assert pipeline.closed


def test_application_session_honors_sink_stop_decision() -> None:
    pipeline = _FakePipeline()
    output_sink = _StoppingSink(store_outputs=True)
    input_handler, session = _initialize_application(
        _application(pipeline),
        pipeline,
        output_sink,
        total_blocks=4,
    )

    while session.step(CanonicalInputs(), output_sink):
        pass

    assert pipeline.generated == [0]
    assert output_sink.output_count == 1


class _NamedInputHandler:
    def __init__(self) -> None:
        self.inputs = CanonicalInputs(values={"camera": {"yaw": 0.25, "pitch": -0.5}})

    def open(self, session_info: object) -> None:
        del session_info

    def current_inputs(self) -> CanonicalInputs:
        return self.inputs

    def close(self) -> None:
        return


def test_input_handler_provides_schema_named_canonical_inputs() -> None:
    schema = CanonicalInputSchema(
        modalities=(
            CanonicalModality(
                name="camera",
                payload_fields=frozenset({"yaw", "pitch"}),
            ),
        )
    )
    inputs = application_module._current_application_inputs(
        _NamedInputHandler(), schema
    )

    assert inputs.values == {"camera": {"yaw": 0.25, "pitch": -0.5}}


def test_application_host_writes_mp4_through_shared_io_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )
    writer_calls: list[dict[str, object]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
        layout: str,
        install_hint: str,
    ) -> Path:
        del install_hint
        writer_calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
            }
        )
        return path

    input_handler = NullInputHandler()
    output_sink = Mp4OutputSink(
        output_path=tmp_path / "out.mp4",
        output_layout="tchw",
        writer=fake_writer,
        move_to_cpu=False,
    )
    artifacts = application_module.run_application(
        "t2v-fake",
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"],
        io_factory=ProvidedIOFactory(input_handler, output_sink),
    )

    assert writer_calls == [
        {
            "shape": (4, 3, 4, 5),
            "path": tmp_path / "out.mp4",
            "fps": 16,
            "layout": "tchw",
        }
    ]
    assert len(artifacts) == 1
    assert artifacts[0].kind == "video/mp4"
    assert artifacts[0].uri == str(tmp_path / "out.mp4")
