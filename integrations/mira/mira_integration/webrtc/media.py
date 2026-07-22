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

"""GPU-resident MIRA media helpers and async MP4 writing."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import nvtx
import torch
from loguru import logger
from torch import Tensor

from flashdreams.serving.webrtc.runtime import WebRTCStepResult
from mira_integration.configs.schema import preview_grid_dimensions


@nvtx.annotate("mira.webrtc.media.normalize_player_chunk")
def normalize_player_chunk(video: Tensor, *, n_players: int) -> Tensor:
    """Return a generated chunk in ``[P,T,C,H,W]`` layout."""
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5 or video.shape[0] != n_players:
        raise ValueError(
            f"Expected [{n_players},T,C,H,W] MIRA output, got {tuple(video.shape)}"
        )
    return video


@nvtx.annotate("mira.webrtc.media.tile_player_video")
def tile_player_video(video: Tensor) -> Tensor:
    """Tile per-player video into one near-square preview image stream."""
    if video.ndim != 5 or video.shape[0] <= 0:
        raise ValueError(f"Expected [P,T,C,H,W] player video, got {tuple(video.shape)}")
    players, frames, channels, height, width = video.shape
    rows, columns = preview_grid_dimensions(players)
    preview = torch.zeros(
        frames,
        channels,
        rows * height,
        columns * width,
        dtype=video.dtype,
        device=video.device,
    )
    for player in range(players):
        row, column = divmod(player, columns)
        preview[
            :,
            :,
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = video[player]
    return preview


@nvtx.annotate("mira.webrtc.media.video_to_uint8_image")
def video_to_uint8_image(video: Tensor) -> Tensor:
    """Convert ``[0,1]`` RGB video to uint8 while preserving device placement."""
    if video.dtype == torch.uint8:
        return video.detach()
    return (
        video.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
    )


@nvtx.annotate("mira.webrtc.media.copy_tensor_to_host")
def copy_tensor_to_host(tensor: Tensor) -> Tensor:
    """Copy a contiguous tensor to host memory without using the render thread."""
    tensor = tensor.detach().contiguous()
    if tensor.device.type == "cpu":
        return tensor
    if tensor.device.type != "cuda":
        return tensor.cpu()

    with torch.cuda.device(tensor.device):
        host = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        stream = torch.cuda.Stream(device=tensor.device)
        with torch.cuda.stream(stream):
            host.copy_(tensor, non_blocking=True)
        stream.synchronize()
    return host


@nvtx.annotate("mira.webrtc.media.video_chunk_to_thwc_uint8")
def video_chunk_to_thwc_uint8(video_chunk: Tensor) -> np.ndarray:
    """Copy one ``[T,C,H,W]`` GPU/CPU chunk into ``[T,H,W,C]`` host uint8."""
    if video_chunk.ndim != 4 or video_chunk.shape[1] != 3:
        raise ValueError(
            f"Expected [T,3,H,W] RGB chunk, got {tuple(video_chunk.shape)}"
        )
    thwc = video_to_uint8_image(video_chunk).permute(0, 2, 3, 1).contiguous()
    return np.ascontiguousarray(copy_tensor_to_host(thwc).numpy())


@nvtx.annotate("mira.webrtc.media.video_chunk_to_rgb_frames")
def video_chunk_to_rgb_frames(video_chunk: Tensor) -> list[np.ndarray]:
    """Convert one ``[T,C,H,W]`` chunk to host frames for WebRTC."""
    return [np.ascontiguousarray(frame) for frame in video_chunk_to_thwc_uint8(video_chunk)]


@nvtx.annotate("mira.webrtc.media.configure_media_ffmpeg")
def configure_media_ffmpeg(media: Any) -> None:
    """Use PATH FFmpeg or imageio's bundled binary for portable MP4 output."""
    if media.video_is_available():
        return
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing the MIRA demo requires FFmpeg. Install a system FFmpeg or "
            "run `uv pip install imageio-ffmpeg`."
        ) from exc
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    if not media.video_is_available():
        raise RuntimeError("The imageio-ffmpeg executable could not be located.")


@dataclass(kw_only=True)
class MiraMp4Writer:
    """Async sink that copies generated GPU chunks and writes one MP4."""

    output_dir: Path
    runner_name: str
    fps: int
    n_players: int
    stats_history: list[dict[str, float | int]] = field(default_factory=list)
    _frames: list[np.ndarray] = field(default_factory=list, init=False)
    _queue: asyncio.Queue[WebRTCStepResult | None] = field(init=False)
    _worker: asyncio.Task[None] | None = field(default=None, init=False)
    _media: Any = field(default=None, init=False)

    @nvtx.annotate("MiraMp4Writer.__aenter__")
    async def __aenter__(self) -> MiraMp4Writer:
        try:
            import mediapy as media
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Writing the MIRA demo requires the FlashDreams runners extra: "
                "run `uv sync --extra runners`."
            ) from exc
        configure_media_ffmpeg(media)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._media = media
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._consume())
        return self

    @nvtx.annotate("MiraMp4Writer.__aexit__")
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        await self.close(write_video=exc_type is None)

    @nvtx.annotate("MiraMp4Writer.push")
    async def push(self, result: WebRTCStepResult) -> None:
        """Queue a rendered chunk for asynchronous MP4 materialization."""
        if result.stats is not None:
            self.stats_history.append(
                {"autoregressive_index": result.chunk_index, **result.stats}
            )
        await self._queue.put(result)

    @nvtx.annotate("MiraMp4Writer.close")
    async def close(self, *, write_video: bool = True) -> None:
        """Drain pending chunks, then write MP4 and timings."""
        worker = self._worker
        if worker is not None:
            await self._queue.put(None)
            await worker
            self._worker = None
        if write_video and self._frames:
            video = np.concatenate(self._frames, axis=0)
            video_path = self.output_dir / f"{self.runner_name}.mp4"
            await asyncio.to_thread(
                self._media.write_video,
                str(video_path),
                video,
                fps=self.fps,
            )
            logger.info(
                f"[{self.runner_name}] wrote {video.shape} -> {video_path.resolve()}"
            )
        stats_path = self.output_dir / f"stats_{self.runner_name}.json"
        stats_path.write_text(json.dumps(self.stats_history, indent=2))
        logger.info(f"[{self.runner_name}] wrote timings -> {stats_path.resolve()}")

    @nvtx.annotate("MiraMp4Writer._consume")
    async def _consume(self) -> None:
        while True:
            result = await self._queue.get()
            if result is None:
                return
            frames = await asyncio.to_thread(self._prepare_chunk, result.video_chunk)
            self._frames.append(frames)

    @nvtx.annotate("MiraMp4Writer._prepare_chunk")
    def _prepare_chunk(self, video_chunk: Tensor) -> np.ndarray:
        preview = tile_player_video(
            normalize_player_chunk(video_chunk, n_players=self.n_players)
        )
        return video_chunk_to_thwc_uint8(preview)


__all__ = [
    "MiraMp4Writer",
    "configure_media_ffmpeg",
    "copy_tensor_to_host",
    "normalize_player_chunk",
    "tile_player_video",
    "video_chunk_to_rgb_frames",
    "video_chunk_to_thwc_uint8",
    "video_to_uint8_image",
]
