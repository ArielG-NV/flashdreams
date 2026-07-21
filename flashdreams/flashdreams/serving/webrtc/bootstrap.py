# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process bootstrap for the single-session WebRTC demo servers."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger

from flashdreams.serving.webrtc.runtime import WebRTCServerLifecycle


@dataclass(frozen=True, slots=True)
class WebRTCDistributedContext:
    """CUDA/distributed launch context for a WebRTC demo server."""

    device: torch.device
    world_rank: int
    world_size: int


def configure_logging(*, world_rank: int | None = None) -> None:
    from flashdreams.core.distributed import configure_loguru_for_distributed

    configure_loguru_for_distributed(world_rank=world_rank)
    for logger_name in ("aioice", "aioice.ice", "aiortc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _distributed_init() -> None:
    from flashdreams.core.distributed import init as distributed_init

    distributed_init()


def initialize_cuda_distributed(
    *,
    default_device: str | torch.device = "cuda:0",
    distributed_init_fn: Callable[[], None] | None = None,
    configure_logging_fn: Callable[..., None] = configure_logging,
    torch_module: Any = torch,
    dist_module: Any = dist,
) -> WebRTCDistributedContext:
    """Initialize CUDA and optional torch.distributed for WebRTC serving."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for inference in the WebRTC server.")

    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if has_rank != has_world_size:
        raise RuntimeError(
            "Distributed launch expects both RANK and WORLD_SIZE to be set."
        )

    distributed_launch = has_rank and has_world_size
    if distributed_launch:
        if distributed_init_fn is None:
            distributed_init_fn = _distributed_init
        distributed_init_fn()
        world_rank = dist_module.get_rank()
        world_size = dist_module.get_world_size()
    else:
        world_rank = 0
        world_size = 1

    device_count = torch_module.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA device count must be >= 1 for inference.")
    if distributed_launch:
        local_rank = world_rank % device_count
        torch_device = torch_module.device(f"cuda:{local_rank}")
    else:
        torch_device = torch_module.device(default_device)
        if torch_device.type != "cuda":
            raise RuntimeError(
                f"CUDA device is required for inference, got {torch_device}."
            )
        if torch_device.index is None:
            torch_device = torch_module.device("cuda:0")
    torch_module.cuda.set_device(torch_device)
    configure_logging_fn(world_rank=world_rank)
    return WebRTCDistributedContext(
        device=torch_device,
        world_rank=world_rank,
        world_size=world_size,
    )


def run_webrtc_server(
    *,
    world_rank: int,
    session_manager: WebRTCServerLifecycle,
    app: web.Application | None,
    host: str,
    port: int,
) -> None:
    """Serve on rank 0, idle on worker ranks, then tear the runtime down."""
    if world_rank == 0:
        if app is None:
            raise ValueError("Rank 0 requires an aiohttp app to serve.")
        try:
            web.run_app(app, host=host, port=port)
        finally:
            session_manager.send_exit_signal()
    else:
        try:
            session_manager.wait_for_termination()
        except KeyboardInterrupt:
            logger.warning("Worker rank interrupted, shutting down.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()
        logger.info("[Rank {}] Destroying process group", world_rank)
        dist.destroy_process_group()
