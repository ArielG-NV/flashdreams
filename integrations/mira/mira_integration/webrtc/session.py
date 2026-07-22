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

"""MIRA runtime adapter for the shared WebRTC serving backend."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import nvtx
import torch

from flashdreams.serving.webrtc.runtime import WebRTCStepResult
from mira_integration.configs.schema import MiraModelMetadata, MiraWebRTCModelConfig
from mira_integration.pipeline import MiraCache, MiraPipeline, MiraPipelineConfig

MIRA_VIDEO_HEIGHT = 288
"""Pixel height emitted by the published 9-token MIRA latent grid."""

MIRA_VIDEO_WIDTH = 512
"""Pixel width emitted by the published 16-token MIRA latent grid."""

MIRA_FRAMES_PER_CHUNK = 2
"""Pixel frames decoded from each MIRA autoregressive latent."""

_T = TypeVar("_T")


@dataclass(kw_only=True)
class MiraRuntimeConfig:
    """Configuration for one persistent MIRA WebRTC runtime."""

    model_config: MiraWebRTCModelConfig
    """Manifest-selected metadata and generated native pipeline config."""

    device: str = "cuda:0"
    """Torch device used for model inference."""

    seed: int = 0
    """Torch RNG seed restored at the start of every browser session."""

    fps: int = 60
    """WebRTC playback and keyboard-resampling frame rate."""

    video_height: int = MIRA_VIDEO_HEIGHT
    """Fixed MIRA output height reported to the browser."""

    video_width: int = MIRA_VIDEO_WIDTH
    """Fixed MIRA output width reported to the browser."""

    frames_per_chunk: int = MIRA_FRAMES_PER_CHUNK
    """Pixel frames expected from each generated chunk."""

    n_diffusion_steps: int = 2
    """Sampler steps used for each generated latent frame."""

    warmup_chunks: int = 2
    """Synthetic chunks generated before accepting browser sessions."""

    warmup_timeout_s: float = 600.0
    """Maximum time allowed for the loopback warmup session."""

    def __post_init__(self) -> None:
        metadata = self.model_config.metadata
        if (self.video_height, self.video_width) != (
            metadata.video_height,
            metadata.video_width,
        ):
            raise ValueError(
                f"{metadata.display_name} output is fixed at "
                f"{metadata.video_width}x{metadata.video_height}."
            )
        if self.frames_per_chunk != metadata.frames_per_chunk:
            raise ValueError(
                f"{metadata.display_name} emits {metadata.frames_per_chunk} "
                "frames per chunk."
            )
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be > 0")
        if self.n_diffusion_steps <= 0:
            raise ValueError("n_diffusion_steps must be > 0")
        if self.warmup_chunks < 0:
            raise ValueError("warmup_chunks must be >= 0")


@nvtx.annotate()
def checkpoint_keys(
    keys: frozenset[str],
    metadata: MiraModelMetadata,
) -> list[str]:
    """Translate normalized browser keys into MIRA checkpoint names."""
    return metadata.checkpoint_keys(keys)


class MiraInferenceRuntime:
    """Run a persistent MIRA pipeline behind the shared async WebRTC manager."""

    def __init__(
        self,
        *,
        config: MiraRuntimeConfig,
        pipeline_factory: Callable[[MiraPipelineConfig], MiraPipeline] | None = None,
    ) -> None:
        self.config = config
        self.model_config = config.model_config
        self._pipeline_factory = pipeline_factory or self._setup_pipeline
        self._pipeline: MiraPipeline | None = None
        self._cache: MiraCache | None = None
        self._autoregressive_index = 0
        self._closed = False
        self._step_lock = asyncio.Lock()
        self._exit_event = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mira-webrtc-runtime",
        )

    @staticmethod
    @nvtx.annotate()
    def _setup_pipeline(config: MiraPipelineConfig) -> MiraPipeline:
        pipeline = config.setup()
        if not isinstance(pipeline, MiraPipeline):
            raise TypeError("MIRA WebRTC requires a MiraPipeline instance.")
        return pipeline

    @nvtx.annotate()
    async def initialize(self) -> None:
        """Construct the MIRA pipeline on its dedicated runtime thread."""
        if self._closed:
            raise RuntimeError("MIRA runtime is closed.")
        if self._pipeline is None:
            await self._run_on_runtime_thread(self._initialize_sync)

    @nvtx.annotate()
    async def reset_for_new_session(self) -> None:
        """Reset the cache and RNG for a new browser session."""
        if self._closed:
            raise RuntimeError("MIRA runtime is closed.")
        await self._run_on_runtime_thread(self._reset_sync)

    def peek_steady_chunk_num_frames(self) -> int:
        """Return the fixed number of frames in a MIRA output chunk."""
        return self.config.frames_per_chunk

    def peek_next_chunk_num_frames(self) -> int:
        """Return the frame count expected from the next MIRA step."""
        return self.config.frames_per_chunk

    @nvtx.annotate()
    async def generate_chunk(
        self,
        *,
        player_keys: tuple[frozenset[str] | None, ...],
    ) -> WebRTCStepResult:
        """Generate one synchronized chunk from all four held-key states."""
        async with self._step_lock:
            if self._closed:
                raise RuntimeError("MIRA runtime is closed.")
            return await self._run_on_runtime_thread(
                self._generate_chunk_sync,
                player_keys,
            )

    @nvtx.annotate()
    async def close(self) -> None:
        """Release pipeline state and stop the dedicated runtime thread."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._run_on_runtime_thread(self._close_sync)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._exit_event.set()

    def wait_for_termination(self) -> None:
        """Wait for the rank-zero server to request process termination."""
        self._exit_event.wait()

    def send_exit_signal(self) -> None:
        """Release any worker waiting for server termination."""
        self._exit_event.set()

    @nvtx.annotate()
    async def _run_on_runtime_thread(
        self,
        function: Callable[..., _T],
        *args: Any,
    ) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._runtime_thread_entry,
            function,
            args,
        )

    @nvtx.annotate()
    def _runtime_thread_entry(
        self,
        function: Callable[..., _T],
        args: tuple[Any, ...],
    ) -> _T:
        device = torch.device(self.config.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return function(*args)

    @nvtx.annotate()
    def _initialize_sync(self) -> None:
        self._pipeline = (
            self._pipeline_factory(self.model_config.pipeline)
            .to(device=self.config.device)
            .eval()
        )

    @nvtx.annotate()
    def _reset_sync(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("MIRA runtime is not initialized.")
        torch.manual_seed(self.config.seed)
        if torch.device(self.config.device).type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        self._cache = self._pipeline.initialize_cache(
            n_diffusion_steps=self.config.n_diffusion_steps
        )
        self._autoregressive_index = 0

    @nvtx.annotate()
    def _generate_chunk_sync(
        self,
        player_keys: tuple[frozenset[str] | None, ...],
    ) -> WebRTCStepResult:
        if self._pipeline is None or self._cache is None:
            raise RuntimeError("MIRA session is not initialized.")
        player_count = self.model_config.metadata.player_count
        if len(player_keys) != player_count:
            raise ValueError(
                f"MIRA expects controls for {player_count} players, "
                f"got {len(player_keys)}."
            )

        with nvtx.annotate("MiraInferenceRuntime.translate_keys"):
            held_keys = [
                None
                if keys is None
                else checkpoint_keys(keys, self.model_config.metadata)
                for keys in player_keys
            ]
        chunk_index = self._autoregressive_index
        with nvtx.annotate("MiraInferenceRuntime.pipeline_generate"):
            video = self._pipeline.generate(
                chunk_index,
                self._cache,
                input=held_keys,
            )
        with nvtx.annotate("MiraInferenceRuntime.pipeline_finalize"):
            stats = self._pipeline.finalize(chunk_index, self._cache)
        with nvtx.annotate("MiraInferenceRuntime.materialize_uint8_cpu"):
            video_uint8 = (
                video.detach()
                .float()
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
            )
        if video_uint8.ndim == 4:
            video_uint8 = video_uint8.unsqueeze(0)
        self._autoregressive_index += 1
        return WebRTCStepResult(
            chunk_index=chunk_index,
            num_frames=int(video_uint8.shape[1]),
            video_chunk=video_uint8,
            stats=stats,
        )

    @nvtx.annotate()
    def _close_sync(self) -> None:
        pipeline = self._pipeline
        self._cache = None
        self._pipeline = None
        if pipeline is not None:
            pipeline.close()
        if (
            torch.device(self.config.device).type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()


__all__ = [
    "MIRA_FRAMES_PER_CHUNK",
    "MIRA_VIDEO_HEIGHT",
    "MIRA_VIDEO_WIDTH",
    "MiraInferenceRuntime",
    "MiraRuntimeConfig",
    "checkpoint_keys",
]
