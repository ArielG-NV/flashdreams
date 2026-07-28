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
import csv
import io
import json
import math
import time
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
    return video.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)


@nvtx.annotate("mira.webrtc.media.copy_tensor_to_host")
def copy_tensor_to_host(tensor: Tensor) -> Tensor:
    """Copy a contiguous tensor to host memory without using the render thread."""
    tensor = tensor.detach().contiguous()
    if tensor.device.type == "cpu":
        return tensor
    if tensor.device.type != "cuda":
        return tensor.cpu()

    with torch.cuda.device(tensor.device):
        producer_stream = torch.cuda.current_stream(tensor.device)
        host = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        stream = torch.cuda.Stream(device=tensor.device)
        stream.wait_stream(producer_stream)
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
    return [
        np.ascontiguousarray(frame) for frame in video_chunk_to_thwc_uint8(video_chunk)
    ]


@nvtx.annotate("mira.webrtc.media.write_video")
def _write_video(path: Path, video: np.ndarray, *, fps: int) -> None:
    """Encode contiguous ``[T,H,W,3]`` RGB frames as an H.264 MP4 with PyAV."""
    import av

    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 video in [T,H,W,3] layout, got {video.shape} "
            f"with dtype {video.dtype}"
        )

    _, height, width, _ = video.shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_array in video:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def _format_runtime_metrics_csv(
    *,
    runner_name: str,
    fps: int,
    model_vram_bytes: int,
    gpu_name: str,
    action_script: str,
    stats_history: list[dict[str, float | int]],
) -> str:
    profiled_chunks = [
        stats
        for stats in stats_history
        if float(stats.get("total_ms", 0)) > 0
        and int(stats.get("frames_per_chunk", 0)) > 0
    ]
    gib = 1024**3
    metrics = {
        "runner": runner_name,
        "target_cap_fps": str(fps),
        "gpu_name": gpu_name,
        "action-script": action_script,
        "model_load_vram_bytes": str(model_vram_bytes),
        "model_load_vram_gib": f"{model_vram_bytes / gib:.3f}",
    }
    if profiled_chunks:
        latencies = [float(stats["total_ms"]) for stats in profiled_chunks]
        per_chunk_fps = [
            1000 * int(stats["frames_per_chunk"]) / float(stats["total_ms"])
            for stats in profiled_chunks
        ]
        per_frame_latencies = [
            float(stats["total_ms"]) / int(stats["frames_per_chunk"])
            for stats in profiled_chunks
        ]
        frames_per_chunk = int(profiled_chunks[0]["frames_per_chunk"])
        total_profiled_frames = sum(
            int(stats["frames_per_chunk"]) for stats in profiled_chunks
        )
        runtime_average_fps = 1000 * total_profiled_frames / sum(latencies)
        one_percent_count = max(1, math.ceil(len(per_chunk_fps) * 0.01))
        runtime_1_percent_lows_fps = float(
            np.mean(
                np.partition(per_chunk_fps, one_percent_count - 1)[:one_percent_count]
            )
        )
        model_p90_latency = np.percentile(latencies, 90)
        runtime_p90_latency = np.percentile(per_frame_latencies, 90)

        metrics.update(
            frames_per_chunk=str(frames_per_chunk),
            model_latency_p90_ms=f"{model_p90_latency:.3f}",
            runtime_average_fps=f"{runtime_average_fps:.3f}",
            runtime_1_percent_lows_fps=f"{runtime_1_percent_lows_fps:.3f}",
            runtime_latency_p90_ms=f"{runtime_p90_latency:.3f}",
        )
    if stats_history:
        latest = stats_history[-1]
        for key in ("mem_alloc_gib", "mem_reserved_gib", "mem_peak_gib"):
            if key in latest:
                metrics[f"runtime_{key}"] = f"{float(latest[key]):.3f}"

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(metrics),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(metrics)
    return output.getvalue()


def _average_runtime_fps(
    stats_history: list[dict[str, float | int]],
) -> float | None:
    """Return generated frames per profiled model second."""
    profiled = [
        stats
        for stats in stats_history
        if float(stats.get("total_ms", 0)) > 0
        and int(stats.get("frames_per_chunk", 0)) > 0
    ]
    if not profiled:
        return None
    total_frames = sum(int(stats["frames_per_chunk"]) for stats in profiled)
    total_ms = sum(float(stats["total_ms"]) for stats in profiled)
    return 1000 * total_frames / total_ms


@nvtx.annotate("mira.webrtc.media.materialize_average_fps_video")
def materialize_average_fps_video(
    chunks: list[np.ndarray],
    *,
    output_fps: int,
    average_fps: float | None,
) -> np.ndarray:
    """Concatenate every generated frame and hold the last frame across gaps.

    Repeated frames are distributed after chunks so playback at ``output_fps``
    represents the measured average generation rate. Generated frames are never
    removed when inference is faster than the requested output rate.
    """
    if output_fps <= 0:
        raise ValueError("output_fps must be > 0")
    if not chunks:
        raise ValueError("chunks must not be empty")

    outputs: list[np.ndarray] = []
    generated_frames = 0
    materialized_frames = 0
    for chunk in chunks:
        outputs.append(chunk)
        generated_frames += len(chunk)
        materialized_frames += len(chunk)
        if average_fps is None or average_fps <= 0:
            continue

        expected_frames = max(
            generated_frames,
            math.ceil(generated_frames * output_fps / average_fps),
        )
        hold_count = expected_frames - materialized_frames
        if hold_count > 0:
            outputs.append(np.repeat(chunk[-1:], hold_count, axis=0))
            materialized_frames += hold_count
    return np.ascontiguousarray(np.concatenate(outputs, axis=0))


@dataclass(kw_only=True)
class MiraMp4Writer:
    """Retain generated frames and write one throughput-aware MP4 at the end."""

    output_dir: Path
    runner_name: str
    fps: int
    n_players: int
    model_vram_bytes: int
    """CUDA allocator growth caused by loading the model."""

    gpu_name: str
    """Name reported by the configured CUDA device."""

    action_script: str
    """Unmodified action script supplied to the runner."""

    stats_history: list[dict[str, float | int]] = field(default_factory=list)
    _chunks: list[Tensor] = field(default_factory=list, init=False)
    _recording_start_time: float = field(default=0.0, init=False)
    """Monotonic clock origin used for runtime metrics."""

    @nvtx.annotate("MiraMp4Writer.__aenter__")
    async def __aenter__(self) -> MiraMp4Writer:
        try:
            import av  # noqa: F401, PLC0415
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Writing the MIRA demo requires PyAV: "
                "run `uv sync --package flashdreams-mira`."
            ) from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._recording_start_time = time.perf_counter()
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
        """Retain a rendered chunk without copying or encoding it."""
        ready_time_s = time.perf_counter() - self._recording_start_time
        stats: dict[str, float | int] = {
            "autoregressive_index": result.chunk_index,
            "frames_per_chunk": result.num_frames,
            "recording_elapsed_ms": ready_time_s * 1000,
            "real_time_budget_ms": result.num_frames * 1000 / self.fps,
        }
        if result.stats is not None:
            stats.update(result.stats)
        self.stats_history.append(stats)
        self._chunks.append(result.video_chunk.detach())

    @nvtx.annotate("MiraMp4Writer.close")
    async def close(self, *, write_video: bool = True) -> None:
        """Encode all collected frames once, then write runtime metrics."""
        if write_video and self._chunks:
            chunks = [
                await asyncio.to_thread(self._prepare_chunk, chunk)
                for chunk in self._chunks
            ]
            video = materialize_average_fps_video(
                chunks,
                output_fps=self.fps,
                average_fps=_average_runtime_fps(self.stats_history),
            )
            video_path = self.output_dir / f"{self.runner_name}.mp4"
            await asyncio.to_thread(
                _write_video,
                video_path,
                video,
                fps=self.fps,
            )
            logger.info(
                f"[{self.runner_name}] wrote {video.shape} -> {video_path.resolve()}"
            )
        stats_path = self.output_dir / f"stats_{self.runner_name}.json"
        stats_path.write_text(json.dumps(self.stats_history, indent=2))
        logger.info(f"[{self.runner_name}] wrote timings -> {stats_path.resolve()}")
        metrics_path = self.output_dir / f"metrics_{self.runner_name}.csv"
        metrics_path.write_text(
            _format_runtime_metrics_csv(
                runner_name=self.output_dir.name,
                fps=self.fps,
                model_vram_bytes=self.model_vram_bytes,
                gpu_name=self.gpu_name,
                action_script=self.action_script,
                stats_history=self.stats_history,
            )
        )
        logger.info(f"[{self.runner_name}] wrote metrics -> {metrics_path.resolve()}")

    @nvtx.annotate("MiraMp4Writer._prepare_chunk")
    def _prepare_chunk(self, video_chunk: Tensor) -> np.ndarray:
        preview = tile_player_video(
            normalize_player_chunk(video_chunk, n_players=self.n_players)
        )
        return video_chunk_to_thwc_uint8(preview)


__all__ = [
    "MiraMp4Writer",
    "copy_tensor_to_host",
    "materialize_average_fps_video",
    "normalize_player_chunk",
    "tile_player_video",
    "video_chunk_to_rgb_frames",
    "video_chunk_to_thwc_uint8",
    "video_to_uint8_image",
]
