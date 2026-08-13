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

from typing import Any

import pytest
from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
    T2VApplicationSession,
)

from flashdreams.demo import NullInputSink, NullOutputSink

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

    def generate(self, *, autoregressive_index: int, cache: object) -> str:
        del cache
        self.generated.append(autoregressive_index)
        return f"frame-{autoregressive_index}"

    def finalize(self, *, autoregressive_index: int, cache: object) -> None:
        del cache
        self.finalized.append(autoregressive_index)

    def close(self) -> None:
        self.closed = True


class _FakePipelineConfig:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> _FakePipeline:
        return self.pipeline


def _application(pipeline: _FakePipeline) -> T2VApplication:
    return T2VApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=_FakePipelineConfig(pipeline),
            total_blocks=4,
            pixel_height=480,
            pixel_width=832,
        )
    )


def test_prompt_is_required() -> None:
    application = _application(_FakePipeline())
    with pytest.raises(ValueError, match="--prompt is required"):
        application.init(
            [],
            NullInputSink(),
            NullOutputSink(),
        )


def test_application_session_sends_generated_chunks_to_opaque_sink() -> None:
    pipeline = _FakePipeline()
    input_src = NullInputSink()
    output_sink = NullOutputSink(store_outputs=True)
    application = _application(pipeline)
    application.init(
        [
            "--prompt",
            "A waterfall",
            "--total-blocks",
            "2",
            "--device",
            "cpu",
        ],
        input_src,
        output_sink,
    )

    session = application.createSession(input_src, output_sink)
    assert isinstance(session, T2VApplicationSession)
    session.init()
    session.generate(input_src, output_sink)
    session.close()

    assert pipeline.device == "cpu"
    assert pipeline.cache_kwargs == {
        "text": ["A waterfall"],
        "image": None,
        "height": 60,
        "width": 104,
    }
    assert pipeline.generated == [0, 1]
    assert pipeline.finalized == [0, 1]
    assert output_sink.outputs == ["frame-0", "frame-1"]
    assert pipeline.closed
