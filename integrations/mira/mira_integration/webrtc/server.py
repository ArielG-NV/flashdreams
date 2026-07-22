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

"""Aiohttp server for interactive MIRA WebRTC inference."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import replace
from importlib.resources import as_file, files
from pathlib import Path

import nvtx
from aiohttp import web
from loguru import logger

from flashdreams.infra.config import derive_config
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
    run_webrtc_server,
)
from flashdreams.serving.webrtc.server import (
    SESSION_MANAGER_KEY,
    SessionBusyError,
    WebRTCSessionManager,
    create_packaged_webrtc_app,
    create_webrtc_app,
)
from flashdreams.serving.webrtc.server import (
    close_package_resources as _close_package_resources,
)
from mira_integration.configs import load_demo_config
from mira_integration.webrtc.room import MiraMultiplayerSessionManager
from mira_integration.webrtc.session import (
    MiraRuntimeConfig,
)

WEB_DIR_RESOURCE = files("mira_integration.webrtc").joinpath("web")
"""Packaged browser assets served by the MIRA WebRTC app."""


@nvtx.annotate()
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse MIRA WebRTC server arguments.

    Args:
        argv: Explicit argument sequence; ``None`` reads ``sys.argv``.

    Returns:
        Parsed server arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "MIRA WebRTC server: stream a configured multiplayer world model "
            "to browsers in one shared room."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--n-diffusion-steps", type=int, default=None)
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile-network",
        dest="compile_network",
        action="store_true",
        default=True,
        help="Compile the MIRA DiT before CUDA graph capture (default).",
    )
    compile_group.add_argument(
        "--no-compile-network",
        dest="compile_network",
        action="store_false",
        help="Run the MIRA DiT eagerly.",
    )
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--cuda-graph",
        dest="use_cuda_graph",
        action="store_true",
        default=True,
        help="Use CUDA graphs for steady-state MIRA DiT forwards (default).",
    )
    graph_group.add_argument(
        "--no-cuda-graph",
        dest="use_cuda_graph",
        action="store_false",
        help="Disable CUDA graph replay.",
    )
    parser.add_argument("--cuda-graph-warmup-iters", type=int, default=2)
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--warmup-timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


@nvtx.annotate()
def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager,
) -> web.Application:
    """Create the MIRA aiohttp app with packaged browser assets."""
    manager = session_manager

    @nvtx.annotate()
    def configure_multiplayer_routes(app: web.Application) -> None:
        async def model_config(request: web.Request) -> web.StreamResponse:
            room_manager = request.app[SESSION_MANAGER_KEY]
            public_config = getattr(room_manager, "public_config", None)
            if not callable(public_config):
                raise web.HTTPServiceUnavailable(reason="MIRA config is unavailable.")
            return web.json_response(public_config())

        async def room_state(request: web.Request) -> web.StreamResponse:
            room_manager = request.app[SESSION_MANAGER_KEY]
            if not isinstance(room_manager, MiraMultiplayerSessionManager):
                return web.json_response(
                    {"players": [], "capacity": 0, "runtime_ready": True}
                )
            return web.json_response(room_manager.room_state())

        async def player_offer(request: web.Request) -> web.StreamResponse:
            try:
                payload = await request.json()
                sdp = payload["sdp"]
                offer_type = payload["type"]
            except (KeyError, TypeError, ValueError) as exc:
                raise web.HTTPBadRequest(reason="Offer requires sdp and type.") from exc
            room_manager = request.app[SESSION_MANAGER_KEY]
            if not isinstance(room_manager, MiraMultiplayerSessionManager):
                raise web.HTTPNotImplemented(reason="Multiplayer is unavailable.")
            try:
                if "seat" in payload:
                    answer = await room_manager.create_player_answer(
                        seat=int(payload["seat"]),
                        offer_sdp=sdp,
                        offer_type=offer_type,
                    )
                else:
                    answer = await room_manager.create_answer(
                        offer_sdp=sdp,
                        offer_type=offer_type,
                    )
            except SessionBusyError as exc:
                raise web.HTTPConflict(reason=str(exc)) from exc
            except ValueError as exc:
                raise web.HTTPBadRequest(reason=str(exc)) from exc
            return web.json_response(answer)

        app.router.add_get("/api/mira/config", model_config)
        app.router.add_get("/api/mira/room", room_state)
        app.router.add_post("/api/mira/offer", player_offer)

    return create_packaged_webrtc_app(
        web_resource=WEB_DIR_RESOURCE,
        session_manager=manager,
        request_session_url=request_session_url,
        preload_name="MIRA",
        as_file_fn=as_file,
        create_app_fn=create_webrtc_app,
        configure_app=configure_multiplayer_routes,
        cleanup_callback=_close_package_resources,
    )


@nvtx.annotate()
def build_runtime_config(args: argparse.Namespace) -> MiraRuntimeConfig:
    """Build the runtime config from parsed server arguments."""
    model_config = load_demo_config(args.manifest, args.demo)
    model_config = replace(
        model_config,
        pipeline=derive_config(
            model_config.pipeline,
            diffusion_model=dict(
                transformer=dict(
                    compile_network=args.compile_network,
                    use_cuda_graph=args.use_cuda_graph,
                    cuda_graph_warmup_iters=args.cuda_graph_warmup_iters,
                )
            ),
        ),
    )
    n_diffusion_steps = args.n_diffusion_steps
    if n_diffusion_steps is None:
        n_diffusion_steps = model_config.metadata.steps
    if args.cuda_graph_warmup_iters < 0:
        raise ValueError("--cuda-graph-warmup-iters must be >= 0")
    return MiraRuntimeConfig(
        device=args.device,
        model_config=model_config,
        seed=args.seed,
        fps=args.fps,
        n_diffusion_steps=n_diffusion_steps,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
    )


@nvtx.annotate()
def main() -> None:
    """Initialize CUDA and serve the interactive MIRA browser UI."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("MIRA WebRTC supports one GPU only.")

    configure_logging()
    args = parse_args()
    runtime_config = build_runtime_config(args)
    transformer_config = runtime_config.model_config.pipeline.diffusion_model.transformer
    logger.info(
        "MIRA acceleration: compile_network={} cuda_graph={} "
        "cuda_graph_warmup_iters={} n_diffusion_steps={} fps={}",
        transformer_config.compile_network,
        transformer_config.use_cuda_graph,
        transformer_config.cuda_graph_warmup_iters,
        runtime_config.n_diffusion_steps,
        runtime_config.fps,
    )
    distributed_context = initialize_cuda_distributed(default_device=args.device)
    runtime_config.device = str(distributed_context.device)
    session_manager = MiraMultiplayerSessionManager(runtime_config=runtime_config)
    external_ip = get_external_ip()
    app = create_app(
        session_manager=session_manager,
        request_session_url=f"http://{external_ip}:{args.port}/request_session",
    )
    logger.info("Starting MIRA WebRTC on external IP: {}", external_ip)
    run_webrtc_server(
        world_rank=distributed_context.world_rank,
        session_manager=session_manager,
        app=app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
