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

"""CPU contracts for MIRA presentation-time post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from mira_integration.webrtc.media import tile_player_video, untile_player_video
from mira_integration.webrtc.server import parse_args
from mira_integration.webrtc.session import (
    MiraInferenceRuntime,
    MiraRuntimeConfig,
    flashvsr_postprocess_chain,
)

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
)

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _UpscaleBufferConfig(VideoPostProcessorConfig):
    _target: type["_UpscaleBuffer"] = field(default_factory=lambda: _UpscaleBuffer)

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        return VideoSpec(
            height=input_spec.height * 2,
            width=input_spec.width * 2,
            fps=input_spec.fps,
            channels=input_spec.channels,
        )


class _UpscaleBuffer(VideoPostProcessor[_UpscaleBufferConfig]):
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        return _UpscaleBufferSession()


class _UpscaleBufferSession(VideoPostProcessorSession):
    def __init__(self) -> None:
        self._buffer: torch.Tensor | None = None

    def reset(self) -> None:
        self._buffer = None

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        video = chunk.tensor
        self._buffer = (
            video if self._buffer is None else torch.cat((self._buffer, video), dim=0)
        )
        if self._buffer.shape[0] < 3:
            return []
        output = self._buffer[:3]
        self._buffer = self._buffer[3:]
        return [self._upscale(output)]

    def flush(self) -> list[VideoChunk]:
        if self._buffer is None or self._buffer.shape[0] == 0:
            return []
        output = self._buffer
        self._buffer = None
        return [self._upscale(output)]

    @staticmethod
    def _upscale(video: torch.Tensor) -> VideoChunk:
        return VideoChunk(
            tensor=video.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3),
            layout="tchw",
        )


class _FakePipeline:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.restore_calls = 0

    def to(self, **_kwargs: Any) -> _FakePipeline:
        return self

    def eval(self) -> _FakePipeline:
        return self

    def initialize_cache(self, *, n_diffusion_steps: int) -> dict[str, int]:
        return {"steps": n_diffusion_steps}

    def restore_cache(self, cache: object) -> None:
        del cache
        self.restore_calls += 1

    def generate(
        self,
        autoregressive_index: int,
        cache: object,
        input: list[list[str] | None],
    ) -> torch.Tensor:
        del autoregressive_index, cache, input
        self.generate_calls += 1
        value = float(self.generate_calls % 2)
        return torch.full((2, 3, 2, 3), value)

    def finalize(self, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"total_ms": 2.0}

    def close(self) -> None:
        return


def _runtime_config() -> MiraRuntimeConfig:
    metadata = SimpleNamespace(
        player_count=1,
        video_height=2,
        video_width=3,
        frames_per_chunk=2,
        browser_keys=frozenset({"w"}),
        checkpoint_keys=lambda keys: [key.upper() for key in sorted(keys)],
    )
    model_config = SimpleNamespace(
        metadata=metadata,
        pipeline=SimpleNamespace(enable_sync_and_profile=False),
    )
    return MiraRuntimeConfig(
        model_config=model_config,  # ty: ignore[invalid-argument-type]
        device="cpu",
        postprocess=VideoPostprocessChainConfig(processors=(_UpscaleBufferConfig(),)),
    )


@pytest.mark.asyncio
async def test_runtime_upscales_before_returning_and_flushes_tail() -> None:
    pipeline = _FakePipeline()
    config = _runtime_config()
    runtime = MiraInferenceRuntime(
        config=config,
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    try:
        await runtime.initialize()
        await runtime.reset_for_new_session()
        result = await runtime.generate_chunk(player_keys=(frozenset({"w"}),))
        tail = await runtime.flush_postprocess()

        assert pipeline.generate_calls == 2
        assert result.video_chunk.shape == (1, 3, 3, 4, 6)
        assert result.num_frames == 3
        assert result.stats == {"total_ms": 4.0, "native_chunks": 2.0}
        assert tail is not None
        assert tail.video_chunk.shape == (1, 1, 3, 4, 6)
        assert (config.video_height, config.video_width) == (4, 6)
        assert config.frames_per_chunk == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_reset_discards_postprocess_history() -> None:
    pipeline = _FakePipeline()
    runtime = MiraInferenceRuntime(
        config=_runtime_config(),
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    try:
        await runtime.initialize()
        await runtime.reset_for_new_session()
        await runtime.render_next_chunk()
        assert pipeline.generate_calls == 2

        await runtime.reset_for_new_session()
        await runtime.render_next_chunk()
        assert pipeline.generate_calls == 4
        assert pipeline.restore_calls == 1
    finally:
        await runtime.close()


def test_tile_round_trip_preserves_player_views() -> None:
    players = torch.arange(3 * 2 * 3 * 2 * 4).reshape(3, 2, 3, 2, 4)
    tiled = tile_player_video(players)
    restored = untile_player_video(tiled, n_players=3)

    assert torch.equal(restored, players)


def test_webrtc_cli_exposes_flashvsr_toggle() -> None:
    args = parse_args(
        [
            "--manifest",
            "manifest.yaml",
            "--demo",
            "demo",
            "--flashvsr",
        ]
    )
    assert args.flashvsr is True


def test_flashvsr_config_reports_presented_resolution_and_chunk_size() -> None:
    metadata = SimpleNamespace(
        player_count=1,
        video_height=288,
        video_width=512,
        frames_per_chunk=2,
    )
    model_config = SimpleNamespace(metadata=metadata)
    config = MiraRuntimeConfig(
        model_config=model_config,  # ty: ignore[invalid-argument-type]
        postprocess=flashvsr_postprocess_chain(),
    )

    assert (config.video_height, config.video_width) == (512, 1024)
    assert config.frames_per_chunk == 8
