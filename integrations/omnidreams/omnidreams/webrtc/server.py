# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.interactive_drive.cli_args import (
    ExplicitArgTrackingArgumentParser,
    arg_was_explicit,
)
from omnidreams.interactive_drive.config import WorldModelProfileConfig
from omnidreams.interactive_drive.world_model.flashdreams_adapter import (
    _build_pipeline_config,
)
from omnidreams.interactive_drive.world_model.manifest import (
    load_world_model_manifest,
    resolve_world_model_manifest_path,
)
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.webrtc.session import (
    OmnidreamsMultiplayerSessionManager,
    OmnidreamsRuntimeConfig,
    OmnidreamsSessionInput,
    OmnidreamsWebRTCSessionManager,
)

from flashdreams.core.distributed import (
    init as distributed_init,
)
from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.plugins.registry import discover_postprocess_presets
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

WEB_DIR_RESOURCE = files("omnidreams.webrtc").joinpath("web")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _player_devices(value: str) -> tuple[str, ...]:
    devices = tuple(device.strip() for device in value.split(",") if device.strip())
    if not devices:
        raise argparse.ArgumentTypeError(
            "player devices must be a comma-separated list such as cuda:0,cuda:1"
        )
    try:
        for device in devices:
            torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return devices


class _OmnidreamsSessionManager(WebRTCSessionManager, Protocol):
    runtime_config: OmnidreamsRuntimeConfig

    def set_pending_session_input(
        self,
        session_input: OmnidreamsSessionInput,
        *,
        player_id: int | None = None,
    ) -> None: ...

    def player_descriptors(self) -> list[dict[str, object]]: ...

    def player_preview_jpeg(self, player_id: int) -> bytes | None: ...

    def map_geometry(self) -> dict[str, object]: ...

    async def reset_world(self) -> None: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = ExplicitArgTrackingArgumentParser(
        description=(
            "Omnidreams WebRTC server: serves /request_session and streams "
            "single-view WSAD-controlled video chunks over one peer connection."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--pipeline_config_name",
        type=str,
        default="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        choices=sorted(OMNIDREAMS_CONFIGS),
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=None,
        help=(
            "Local WebRTC scene directory containing clipgt/first_image.* "
            "and clipgt/prompt.txt. If omitted, the server downloads and "
            "stages the selected Hugging Face scene."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Omnidreams world-model manifest (YAML). Accepts a path or a "
            "bundled config filename such as example_world_model_perf.yaml. "
            "When set, WebRTC uses the same pipeline perf toggles as the "
            "interactive-drive world-model path."
        ),
    )
    parser.add_argument(
        "--scene-uuid",
        type=str,
        default=None,
        help=(
            "Scene UUID for nvidia/omni-dreams-scenes. Expected dataset asset: "
            "scenes/clipgt-<uuid>[-<variant>].usdz."
        ),
    )
    parser.add_argument(
        "--scene-variant",
        type=str,
        default="default",
        help=(
            "Weather variant to serve: 'default' (clear), 'rain', or 'snow'. "
            "Selects the matching sibling archive and weather prompt."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "-player-count",
        "--player-count",
        "--player_count",
        dest="player_count",
        type=_positive_int,
        default=1,
        help="Number of atomically claimable player slots to expose.",
    )
    parser.add_argument(
        "--player-devices",
        type=_player_devices,
        default=(),
        metavar="DEVICE,...",
        help=(
            "Optional one-device-per-player mapping, for example "
            "'cuda:0,cuda:1'. Players on the same device serialize inference; "
            "players on distinct devices can generate concurrently."
        ),
    )
    parser.add_argument(
        "--live-lobby-previews",
        "--keep-lobby-previews-active",
        dest="live_lobby_previews",
        action="store_true",
        help=(
            "Continuously generate model-backed idle-player thumbnails. "
            "The default lobby uses cached scene snapshots and consumes no "
            "steady-state preview inference."
        ),
    )
    parser.add_argument(
        "--eager-control-chunks",
        action="store_true",
        help=(
            "Sample each control chunk immediately using the latest held state "
            "instead of waiting for its input window to close."
        ),
    )
    parser.add_argument(
        "--single-gpu-multiplayer",
        action="store_true",
        help=(
            "Enable the one-GPU multiplayer preset: eager control chunks and "
            "896x496 video when resolution is otherwise left at its default."
        ),
    )
    parser.add_argument("--video_height", type=int, default=704)
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument(
        "--warmup_chunks",
        type=int,
        default=10,
        help="Number of synthetic startup chunks to generate for kernel autotuning.",
    )
    parser.add_argument(
        "--warmup_timeout_s",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for synthetic startup warmup chunks.",
    )
    parser.add_argument(
        "--debug_serve_hdmaps",
        action="store_true",
        help=(
            "Stream rendered HDMap conditioning frames instead of generated RGB "
            "video. This skips video model generation after initialization."
        ),
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="camera_front_wide_120fov",
    )
    parser.add_argument(
        "--postprocess-preset",
        "--postprocess_preset",
        dest="postprocess_preset",
        default="",
        choices=sorted(discover_postprocess_presets()),
        help=(
            "Video post-process preset for WebRTC sessions. The browser can "
            "only toggle this launched preset before connecting."
        ),
    )
    parser.add_argument(
        "--prefer_sw_encoder",
        action="store_true",
        help=(
            "Prefer the FFmpeg software encoder (aiortc) over the "
            "hardware encoder (PyNvVideoCodec/NVENC H.264). Useful on "
            "hosts where NVENC is unavailable or misbehaving, and for "
            "A/B profiling against the hardware path. Without this flag "
            "the encoder is auto-selected at startup: NVENC when the "
            "driver reports support at the target resolution, aiortc's "
            "software encoder otherwise."
        ),
    )
    return parser.parse_args(argv)


def _get_omnidreams_manager(app: web.Application) -> _OmnidreamsSessionManager:
    return cast(_OmnidreamsSessionManager, app[SESSION_MANAGER_KEY])


async def _postprocess_options(request: web.Request) -> web.StreamResponse:
    manager = _get_omnidreams_manager(request.app)
    configured_preset = manager.runtime_config.postprocess.preset
    presets = [configured_preset] if configured_preset else []
    return web.json_response(
        {
            "default_preset": configured_preset,
            "presets": presets,
        }
    )


async def _session_input(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(reason="Expected JSON session input.") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(reason="Session input must be a JSON object.")
    preset = payload.get("postprocess_preset")
    if not isinstance(preset, str):
        raise web.HTTPBadRequest(
            reason="Session input must include string 'postprocess_preset'."
        )

    player_id = payload.get("player_id")
    if player_id is not None and (
        isinstance(player_id, bool) or not isinstance(player_id, int)
    ):
        raise web.HTTPBadRequest(reason="'player_id' must be an integer.")

    manager = _get_omnidreams_manager(request.app)
    try:
        manager.set_pending_session_input(
            OmnidreamsSessionInput(postprocess_preset=preset),
            player_id=player_id,
        )
    except SessionBusyError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc
    return web.json_response(
        {"postprocess_preset": preset, "player_id": player_id or 1}
    )


async def _players(request: web.Request) -> web.StreamResponse:
    manager = _get_omnidreams_manager(request.app)
    return web.json_response({"players": manager.player_descriptors()})


async def _player_preview(request: web.Request) -> web.StreamResponse:
    try:
        player_id = int(request.match_info["player_id"])
    except ValueError as exc:
        raise web.HTTPBadRequest(reason="Invalid player id.") from exc
    manager = _get_omnidreams_manager(request.app)
    try:
        jpeg = manager.player_preview_jpeg(player_id)
    except ValueError as exc:
        raise web.HTTPNotFound(reason=str(exc)) from exc
    if jpeg is None:
        raise web.HTTPServiceUnavailable(reason="Player preview is warming up.")
    cache_control = (
        "no-store"
        if getattr(manager.runtime_config, "live_lobby_previews", False)
        else "private, max-age=300"
    )
    return web.Response(
        body=jpeg,
        content_type="image/jpeg",
        headers={"Cache-Control": cache_control},
    )


async def _map_geometry(request: web.Request) -> web.StreamResponse:
    manager = _get_omnidreams_manager(request.app)
    return web.json_response(manager.map_geometry())


async def _reset_world(request: web.Request) -> web.StreamResponse:
    manager = _get_omnidreams_manager(request.app)
    await manager.reset_world()
    return web.json_response({"status": "reset"})


async def _game_manager(_: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/request_session")


def _configure_app(app: web.Application) -> None:
    app.router.add_get("/api/postprocess/options", _postprocess_options)
    app.router.add_post("/api/session/input", _session_input)
    app.router.add_get("/api/players", _players)
    app.router.add_get("/api/players/{player_id}/preview.jpg", _player_preview)
    app.router.add_get("/api/map", _map_geometry)
    app.router.add_post("/api/world/reset", _reset_world)
    app.router.add_get("/game-manager", _game_manager)
    app.router.add_get("/", _game_manager)


def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or OmnidreamsWebRTCSessionManager()
    return create_packaged_webrtc_app(
        web_resource=WEB_DIR_RESOURCE,
        session_manager=manager,
        preload_name="Omnidreams",
        request_session_url=request_session_url,
        configure_app=_configure_app,
        as_file_fn=as_file,
        create_app_fn=create_webrtc_app,
        cleanup_callback=_close_package_resources,
    )


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
) -> OmnidreamsRuntimeConfig:
    manifest_path = None
    manifest = None
    pipeline_config = None
    pipeline_config_name = args.pipeline_config_name
    device = args.device
    seed = args.seed
    fps = args.fps
    video_width = args.video_width
    video_height = args.video_height

    manifest_arg = getattr(args, "manifest", None)
    if manifest_arg is not None:
        manifest_path = resolve_world_model_manifest_path(manifest_arg)
        manifest = load_world_model_manifest(manifest_path)
        pipeline_config = _build_pipeline_config(
            manifest,
            profile=WorldModelProfileConfig(),
        )
        pipeline_config_name = str(pipeline_config.name)
        if (
            arg_was_explicit(args, "pipeline_config_name")
            and args.pipeline_config_name != pipeline_config_name
        ):
            raise ValueError(
                "--manifest selects pipeline config "
                f"{pipeline_config_name!r}, but --pipeline_config_name was "
                f"also set to {args.pipeline_config_name!r}."
            )

        if not arg_was_explicit(args, "device"):
            device = manifest.device
        if not arg_was_explicit(args, "seed"):
            seed = manifest.seed_for_every_rollout
        if not arg_was_explicit(args, "fps"):
            fps = manifest.fps
        if not arg_was_explicit(args, "video_width"):
            video_width = manifest.resolution_wh[0]
        if not arg_was_explicit(args, "video_height"):
            video_height = manifest.resolution_wh[1]
    player_count = getattr(args, "player_count", 1)
    player_devices = getattr(args, "player_devices", ())
    single_gpu_multiplayer = getattr(args, "single_gpu_multiplayer", False)
    if single_gpu_multiplayer and player_count < 2:
        raise ValueError("--single-gpu-multiplayer requires --player-count >= 2.")
    if single_gpu_multiplayer and len(set(player_devices)) > 1:
        raise ValueError(
            "--single-gpu-multiplayer cannot be combined with distinct "
            "--player-devices."
        )

    if single_gpu_multiplayer and (video_width, video_height) == (1280, 704):
        video_width, video_height = 896, 496

    return OmnidreamsRuntimeConfig(
        pipeline_config_name=pipeline_config_name,
        pipeline_config=pipeline_config,
        manifest_path=manifest_path,
        scene_dir=args.scene_dir,
        scene_uuid=args.scene_uuid,
        scene_variant=args.scene_variant,
        seed=seed,
        device=device_override or device,
        video_height=video_height,
        video_width=video_width,
        fps=fps,
        player_count=player_count,
        player_devices=player_devices,
        live_lobby_previews=getattr(args, "live_lobby_previews", False),
        pause_lobby_previews_while_active=getattr(
            args,
            "pause_lobby_previews_while_active",
            True,
        ),
        eager_control_chunks=(
            getattr(args, "eager_control_chunks", False)
            or single_gpu_multiplayer
        ),
        camera_name=args.camera_name,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
        debug_serve_hdmaps=args.debug_serve_hdmaps,
        postprocess=VideoPostprocessChainConfig(preset=args.postprocess_preset),
        encoder_backend="default" if args.prefer_sw_encoder else "auto",
    )


def initialize_distributed(
    *,
    default_device: str | torch.device = "cuda:0",
) -> tuple[torch.device, int, int]:
    context = initialize_cuda_distributed(
        default_device=default_device,
        distributed_init_fn=distributed_init,
        configure_logging_fn=configure_logging,
        torch_module=torch,
        dist_module=dist,
    )
    logger.info(
        "Rank {} initialized Omnidreams runtime with context_parallel_size {}",
        context.world_rank,
        context.world_size,
    )
    return context.device, context.world_rank, context.world_size


def _validate_single_view_config(
    config_name: str, pipeline_config: Any | None = None
) -> None:
    pipeline_cfg = pipeline_config or OMNIDREAMS_CONFIGS[config_name]
    transformer_cfg = pipeline_cfg.diffusion_model.transformer
    if not isinstance(transformer_cfg, CosmosTransformerConfig):
        raise TypeError("Omnidreams WebRTC requires a CosmosTransformerConfig.")
    if transformer_cfg.num_views != 1:
        raise ValueError(
            "Omnidreams WebRTC only serves single-view configs; "
            f"{config_name!r} has num_views={transformer_cfg.num_views}."
        )


def main() -> None:
    configure_logging()
    args = parse_args()
    runtime_config = build_runtime_config(args)
    _validate_single_view_config(
        runtime_config.pipeline_config_name,
        runtime_config.pipeline_config,
    )

    runtime_device, world_rank, world_size = initialize_distributed(
        default_device=runtime_config.device
    )
    runtime_config = replace(runtime_config, device=str(runtime_device))
    if runtime_config.player_count > 1 and world_size > 1:
        raise ValueError(
            "Multiplayer WebRTC currently requires one server process; "
            "launch without torchrun/context parallelism."
        )
    session_manager = (
        OmnidreamsWebRTCSessionManager(runtime_config=runtime_config)
        if runtime_config.player_count == 1
        else OmnidreamsMultiplayerSessionManager(runtime_config=runtime_config)
    )
    app = None
    if world_rank == 0:
        external_ip = get_external_ip()
        app = create_app(
            session_manager=session_manager,
            request_session_url=f"http://{external_ip}:{args.port}/game-manager",
        )
        logger.info("Starting on external IP: {}", external_ip)
    run_webrtc_server(
        world_rank=world_rank,
        session_manager=session_manager,
        app=app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
