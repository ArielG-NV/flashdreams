# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import AbstractSet, Any, Callable, TypeVar

import cv2
import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from loguru import logger
from omnidreams.conditioning.conditioning_wrapper import (
    AV_POSITIVE_PROMPT,
    OmnidreamsConditioningState,
    OmnidreamsConditioningWrapper,
    TextPrompt,
)
from omnidreams.conditioning.renderer import load_and_attach_ludus_scene
from omnidreams.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.conditioning.world_scenario.settings import SETTINGS
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.interactive_drive.browser_presenter import NativeHudBrowserPresenter
from omnidreams.interactive_drive.config import RasterConfig, VehicleConfig
from omnidreams.interactive_drive.input.keyboard import (
    KeyboardState,
    command_from_snapshot,
)
from omnidreams.interactive_drive.scene_loader import load_scene_bundle
from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    build_ground_snapper,
    build_map_bounds,
    state_from_initial_pose,
)
from omnidreams.interactive_drive.types import (
    ControlSnapshot,
    PresentedFrame,
    SceneBundle,
)
from omnidreams.scenes import (
    HF_DATASET_BROWSER_URL,
    SCENE_CLIPGT_DIRNAME,
    SCENE_FRAME_SUFFIXES,
    SCENE_FRAMES_DIRNAME,
    SCENE_IMAGE_SUFFIXES,
    SCENE_PROMPT_FILENAME,
    SCENE_VARIANT_DEFAULT,
    hf_hub_download_scene,
    hf_scenes_repo_id,
    prompt_variant_for_scene_variant,
    scenes_cache_root,
)
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.core.distributed.rank_orchestration import (
    RankCoordinator,
    distributed_op,
)
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoPostprocessStream,
)
from flashdreams.plugins.registry import resolve_postprocess_preset
from flashdreams.serving.realtime.media import rgb_array_to_uint8_frames
from flashdreams.serving.webrtc.controls import (
    DRIVING_SUPPORTED_KEYS,
    CameraPoseIntegrator,
    PoseSegment,
)
from flashdreams.serving.webrtc.encoders import (
    EncoderBackend,
    VideoEncoder,
    select_encoder,
)
from flashdreams.serving.webrtc.manager import (
    DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
    WebRTCControlSignal,
    WebRTCStepResult,
    make_webrtc_step_result,
)
from flashdreams.serving.webrtc.server import SessionBusyError

_T = TypeVar("_T")
# Default scene (clear-weather base archive). Weather siblings are selected
# via OmnidreamsRuntimeConfig.scene_variant / the server's --scene-variant.
DEFAULT_WEBRTC_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
# Back-compat aliases for ``omnidreams.scenes`` constants used by external imports.
WEBRTC_SCENES_HF_BROWSER_URL = HF_DATASET_BROWSER_URL
WEBRTC_SCENE_IMAGE_SUFFIXES = SCENE_IMAGE_SUFFIXES


def _resolve_cuda_device(device_spec: str | torch.device) -> torch.device:
    """Resolve a device spec, filling in the active CUDA index when unspecified."""
    device = torch.device(device_spec)
    if device.type == "cuda" and device.index is None:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cuda:0"
        )
    return device


def _choose_existing_asset(
    directory: Path,
    *,
    exact_name: str | None = None,
    fallback_stems: tuple[str, ...] = (),
    fallback_prefixes: tuple[str, ...] = (),
    allowed_suffixes: AbstractSet[str] | None = None,
    preferred_stems: tuple[str, ...] = (),
) -> Path | None:
    if not directory.is_dir():
        return None

    if exact_name is not None:
        exact_path = directory / exact_name
        if exact_path.is_file() and (
            allowed_suffixes is None or exact_path.suffix.lower() in allowed_suffixes
        ):
            return exact_path

    candidates = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        if (
            path.stem in preferred_stems
            or path.stem in fallback_stems
            or any(path.stem.startswith(f"{prefix}-") for prefix in fallback_prefixes)
        ):
            candidates.append(path)

    if not candidates:
        return None

    preferred_order = {stem: index for index, stem in enumerate(preferred_stems)}
    return sorted(
        candidates,
        key=lambda path: (
            preferred_order.get(path.stem, len(preferred_order)),
            path.name,
        ),
    )[0]


def _camera_name_candidates(camera_name: str) -> tuple[str, ...]:
    """Colon/underscore spellings of ``camera_name`` (dataset uses underscores)."""
    underscore = camera_name.replace(":", "_")
    colon = camera_name.replace("_", ":")
    return tuple(dict.fromkeys((camera_name, underscore, colon)))


def _first_frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (2**63 - 1, path.name)


def _resolve_webrtc_first_frame(clipgt_dir: Path, camera_name: str) -> Path | None:
    """Earliest GT frame under ``clipgt/frames/<camera>/``, else ``None``.

    ``None`` when the bundle ships no such frames, so the caller can fall back
    to ``first_image.*``.
    """
    frames_root = clipgt_dir / SCENE_FRAMES_DIRNAME
    if not frames_root.is_dir():
        return None
    candidate_dirs = [
        frames_root / name
        for name in _camera_name_candidates(camera_name)
        if (frames_root / name).is_dir()
    ]
    if not candidate_dirs:
        # Fall back to any single camera directory present.
        candidate_dirs = [
            path for path in sorted(frames_root.iterdir()) if path.is_dir()
        ]
    for directory in candidate_dirs:
        frames = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SCENE_FRAME_SUFFIXES
        ]
        if frames:
            return sorted(frames, key=_first_frame_sort_key)[0]
    return None


def _resolve_webrtc_scene_assets(
    scene_dir: Path,
    *,
    prompt_filename: str,
    clipgt_dirname: str,
    camera_name: str = "camera_front_wide_120fov",
    variant: str = SCENE_VARIANT_DEFAULT,
) -> tuple[Path, Path, Path]:
    missing_assets = []
    clipgt_dir = scene_dir / clipgt_dirname
    if not clipgt_dir.is_dir():
        missing_assets.append(str(scene_dir / clipgt_dirname))
        clipgt_dir = None

    # Prefer the GT camera frame; fall back to ``first_image.*`` for bundles
    # with no per-camera frames.
    first_frame_path = (
        None
        if clipgt_dir is None
        else _resolve_webrtc_first_frame(clipgt_dir, camera_name)
    )
    if first_frame_path is None and clipgt_dir is not None:
        first_frame_path = _choose_existing_asset(
            clipgt_dir,
            fallback_stems=("first_image_1",),
            allowed_suffixes=WEBRTC_SCENE_IMAGE_SUFFIXES,
            preferred_stems=("first_image",),
        )
    if first_frame_path is None:
        missing_assets.append(
            f"frames/<camera>/*.jpeg or first_image.* under {clipgt_dir}/"
        )

    # Prompt matching the weather variant (``promptN.txt``); fall back to a
    # bare ``prompt.txt`` for older bundles.
    weather_prompt_stem = f"prompt{prompt_variant_for_scene_variant(variant)}"
    prompt_path = (
        None
        if clipgt_dir is None
        else _choose_existing_asset(
            clipgt_dir,
            fallback_stems=("prompt1", "prompt2", "prompt3", "prompt"),
            allowed_suffixes={".txt"},
            preferred_stems=(weather_prompt_stem, "prompt"),
        )
    )
    if prompt_path is None:
        missing_assets.append(f"{prompt_filename} under {clipgt_dir}/")

    if missing_assets:
        raise FileNotFoundError(
            "Missing Omnidreams WebRTC scene assets: " + ", ".join(missing_assets)
        )

    assert clipgt_dir is not None
    assert first_frame_path is not None
    assert prompt_path is not None
    return clipgt_dir, first_frame_path, prompt_path


def _safe_extract_zip(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(source) as zf:
        for member in zf.infolist():
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or not member_path.parts
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise ValueError(
                    f"Unsafe archive member in {source}: {member.filename}"
                )
            target = destination / Path(*member_path.parts)
            target_resolved = target.resolve()
            if destination_root != target_resolved and destination_root not in (
                target_resolved.parents
            ):
                raise ValueError(
                    f"Archive member escapes destination: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _variant_dir_suffix(variant: str | None) -> str:
    """Cache subdir / filename suffix for ``variant`` (``""`` for default)."""
    slug = (variant or SCENE_VARIANT_DEFAULT).strip()
    return "" if slug in ("", SCENE_VARIANT_DEFAULT) else f"-{slug}"


def _extract_local_webrtc_scene_if_needed(
    scene_dir: Path,
    *,
    scene_uuid: str | None,
    variant: str = SCENE_VARIANT_DEFAULT,
    clipgt_dirname: str,
) -> Path:
    """Extract the ``scene_uuid`` (+ variant) archive into the local layout."""
    if scene_uuid is None:
        return scene_dir

    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be non-empty when provided."
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")

    suffix = _variant_dir_suffix(variant)
    expected_names = (
        f"clipgt-{scene_uuid}{suffix}.usdz",
        f"{scene_uuid}{suffix}.usdz",
    )
    archive_path = _choose_existing_asset(scene_dir, exact_name=expected_names[0]) or (
        _choose_existing_asset(scene_dir, exact_name=expected_names[1])
    )
    if archive_path is None:
        # Prefer the variant suffix but accept the base archive too.
        archive_path = _choose_existing_asset(
            scene_dir,
            fallback_prefixes=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
            allowed_suffixes={".usdz"},
            preferred_stems=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
        )
    if archive_path is None:
        raise FileNotFoundError(
            "scene_uuid is set but no local USDZ archive was found in "
            f"{scene_dir}. Expected one of: {', '.join(expected_names)}."
        )

    normalized_scene_dir = scene_dir / f"{scene_uuid}{suffix}"
    normalized_clipgt_root = normalized_scene_dir / clipgt_dirname
    _safe_extract_zip(archive_path, normalized_clipgt_root)
    return normalized_scene_dir


def _resolve_game_scene_archive(
    scene_source: Path,
    *,
    scene_uuid: str | None,
    variant: str,
) -> Path:
    """Resolve the intact USDZ required by interactive-drive game physics."""
    if scene_source.is_file() and scene_source.suffix.lower() == ".usdz":
        return scene_source
    if not scene_source.is_dir():
        raise FileNotFoundError(
            f"WebRTC game-mode scene source does not exist: {scene_source}"
        )

    suffix = _variant_dir_suffix(variant)
    preferred_stems: tuple[str, ...]
    if scene_uuid:
        preferred_stems = (
            f"clipgt-{scene_uuid}{suffix}",
            f"{scene_uuid}{suffix}",
            f"clipgt-{scene_uuid}",
            scene_uuid,
        )
    else:
        preferred_stems = ()
    archive = _choose_existing_asset(
        scene_source,
        fallback_prefixes=preferred_stems,
        allowed_suffixes={".usdz"},
        preferred_stems=preferred_stems,
    )
    if archive is None:
        archives = sorted(scene_source.glob("*.usdz"))
        if len(archives) == 1:
            return archives[0]
        raise FileNotFoundError(
            "--game-mode requires an intact USDZ scene archive; none could be "
            f"resolved under {scene_source}. Pass --scene-uuid for a local scene root."
        )
    return archive


def _ensure_hf_webrtc_scene_synced(
    scene_uuid: str,
    *,
    variant: str = SCENE_VARIANT_DEFAULT,
    prompt_filename: str = SCENE_PROMPT_FILENAME,
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME,
) -> Path:
    """Stage an HF scene variant into the WebRTC cache layout.

    Downloads ``scenes/clipgt-<uuid>[-<variant>].usdz`` and extracts it under
    ``FLASHDREAMS_CACHE_DIR/omnidreams-scenes/<uuid>[-<variant>]/clipgt/``. The
    per-uuid+variant directory coexists with the desktop demo's archive files
    in the same root.
    """
    del prompt_filename  # accepted for call-site symmetry; assets resolved later
    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be set."
    suffix = _variant_dir_suffix(variant)
    cache_root = scenes_cache_root()
    scene_dir = cache_root / f"{scene_uuid}{suffix}"
    lock_path = cache_root / ".locks" / f"{scene_uuid}{suffix}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        archive_path = hf_hub_download_scene(scene_uuid, variant)
        _safe_extract_zip(archive_path, scene_dir / clipgt_dirname)

    logger.info(
        "Synced Omnidreams WebRTC scene {} (variant {}) from Hugging Face ({}) to {}",
        scene_uuid,
        variant,
        hf_scenes_repo_id(),
        scene_dir,
    )
    return scene_dir


def _summarize_sdp_candidates(sdp: str) -> str:
    candidates = [
        line.removeprefix("a=candidate:")
        for line in sdp.splitlines()
        if line.startswith("a=candidate:")
    ]
    if not candidates:
        return "0 candidates"

    protocols: dict[str, int] = {}
    addresses: set[str] = set()
    endpoints: list[str] = []
    for candidate in candidates:
        parts = candidate.split()
        if len(parts) >= 5:
            protocols[parts[2].lower()] = protocols.get(parts[2].lower(), 0) + 1
            addresses.add(parts[4])
        if len(parts) >= 6:
            endpoints.append(f"{parts[2].lower()}://{parts[4]}:{parts[5]}")
    protocol_summary = ",".join(
        f"{key}={value}" for key, value in sorted(protocols.items())
    )
    address_summary = ",".join(sorted(addresses)[:8])
    if len(addresses) > 8:
        address_summary += f",+{len(addresses) - 8} more"
    endpoint_summary = ",".join(endpoints[:12])
    if len(endpoints) > 12:
        endpoint_summary += f",+{len(endpoints) - 12} more"
    return (
        f"{len(candidates)} candidates protocols=[{protocol_summary}] "
        f"addresses=[{address_summary}] endpoints=[{endpoint_summary}]"
    )


def _link_or_copy_file(source: Path, target: Path) -> None:
    """Stage a file efficiently without requiring Windows symlink privileges."""
    try:
        os.symlink(source, target)
        return
    except OSError:
        pass

    try:
        os.link(source, target)
        return
    except OSError:
        shutil.copy2(source, target)


class OmnidreamsRuntimeError(RuntimeError):
    """Raised when the Omnidreams WebRTC runtime is used incorrectly."""


@dataclass(slots=True)
class OmnidreamsRuntimeConfig:
    pipeline_config_name: str = (
        "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    )
    pipeline_config: Any | None = None
    manifest_path: Path | None = None
    scene_dir: Path | None = None
    scene_uuid: str | None = None
    # Weather variant slug (default/rain/snow): picks the sibling USDZ + prompt.
    scene_variant: str = SCENE_VARIANT_DEFAULT
    seed: int | None = 42
    device: str = "cuda:0"
    video_height: int = 704
    video_width: int = 1280
    fps: int = 30
    camera_name: str = "camera_front_wide_120fov"
    prompt_filename: str = SCENE_PROMPT_FILENAME
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME
    move_speed_per_s: float = 6.0
    rotate_speed_rad_per_s: float = float(np.deg2rad(35.0))
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0
    debug_serve_hdmaps: bool = False
    game_mode: bool = False
    physics_active_radius_m: float = 96.0
    server_side_hud: bool = True
    postprocess: VideoPostprocessChainConfig = field(
        default_factory=VideoPostprocessChainConfig
    )
    # Video encoder selection. ``"auto"`` prefers NVENC when the driver
    # reports support at the target resolution (Stage-1 probe via
    # ``PyNvVideoCodec.GetEncoderCaps``) and falls back to aiortc's
    # software encoder otherwise. ``"nvenc"`` fails startup if NVENC
    # cannot be initialized. ``"default"`` skips the probe entirely.
    encoder_backend: EncoderBackend = "auto"
    encoder_bitrate_bps: int = 6_000_000
    encoder_gop: int = 30


@dataclass(frozen=True, slots=True)
class OmnidreamsSessionInput:
    """Browser-selectable settings applied to the next WebRTC rollout."""

    postprocess_preset: str | None = None
    """Launched preset selection; ``None`` keeps the CLI default and ``""`` disables it."""


def _validate_requested_postprocess_preset(
    *, requested_preset: str, configured_preset: str
) -> None:
    if not configured_preset:
        raise ValueError(
            "Post-processing is not enabled for this server; restart with "
            "--postprocess-preset to make a preset available."
        )
    if requested_preset != configured_preset:
        raise ValueError(
            "Post-processing preset must match the launched preset "
            f"{configured_preset!r}; got {requested_preset!r}."
        )
    resolve_postprocess_preset(requested_preset)


class OmnidreamsInferenceRuntime:
    """Single-scene, single-view Omnidreams runtime for WebRTC control."""

    def __init__(self, config: OmnidreamsRuntimeConfig | None = None) -> None:
        self.config = config or OmnidreamsRuntimeConfig()
        self.MASTER_RANK = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        control_device = _resolve_cuda_device(self.config.device)

        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.autoregressive_index = 0

        self._device: torch.device | None = None
        self._wrapper: OmnidreamsConditioningWrapper | None = None
        self._state: OmnidreamsConditioningState | None = None
        self._renderer: Any | None = None
        self._scene_data: Any | None = None
        self._initial_rgb_frames: torch.Tensor | None = None
        self._text_prompts: list[TextPrompt] | None = None
        self._camera_to_rig: torch.Tensor | None = None
        self._initial_ego_pose: np.ndarray | None = None
        self._next_timestamp_us: int = 0
        self._postprocess_stream: VideoPostprocessStream | None = None
        self._postprocess_preset = self.config.postprocess.preset
        self._native_hud: NativeHudBrowserPresenter | None = None
        self._native_hud_keyboard: KeyboardState | None = None
        self._native_hud_frames: list[np.ndarray] = []
        self._native_hud_keys: set[str] = set()
        self._game_scene: SceneBundle | None = None
        self._game_simulation: EgoVehicleKinematics | None = None
        self._closed = False
        self._clipgt_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        # Selected once at initialization; the concrete backend is chosen
        # by ``select_encoder`` based on ``config.encoder_backend`` and
        # the driver's ``GetEncoderCaps`` response at
        # ``config.video_width`` / ``config.video_height``.
        self._video_encoder: VideoEncoder | None = None
        # Pin every blocking runtime call to one OS thread: Omnidreams' CUDA
        # graph capture/replay state is thread-local, so spreading calls across
        # workers (e.g. asyncio.to_thread) crashes capture after a few chunks.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omnidreams-webrtc-runtime",
        )

        self._step_lock = asyncio.Lock()
        self.rank_coordinator = RankCoordinator(
            device=control_device,
            signal_type=WebRTCControlSignal,
            is_master=self.is_master,
            master_rank=self.MASTER_RANK,
        )
        self.rank_coordinator.register_distributed_ops(self)

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    @property
    def postprocess_preset(self) -> str:
        """Preset active for the current rollout, or an empty string when off."""
        return self._postprocess_preset

    @property
    def video_encoder(self) -> VideoEncoder:
        """Return the encoder selected at :meth:`initialize` time."""
        if self._video_encoder is None:
            raise OmnidreamsRuntimeError(
                "Video encoder is not initialized; call runtime.initialize() first."
            )
        return self._video_encoder

    def wait_for_termination(self) -> None:
        self.rank_coordinator.worker_loop(exit_signal=WebRTCControlSignal.EXIT)

    def send_exit_signal(self) -> None:
        if self.is_master:
            self.rank_coordinator.send_exit(exit_signal=WebRTCControlSignal.EXIT)

    async def initialize(self) -> None:
        if self._wrapper is not None:
            return
        await self._run_on_runtime_thread(self._initialize_sync_all_ranks)

    async def reset_for_new_session(
        self, session_input: OmnidreamsSessionInput | None = None
    ) -> None:
        if self._closed:
            raise OmnidreamsRuntimeError("Runtime is closed.")
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        await self._run_on_runtime_thread(
            self._reset_rollout_sync_all_ranks,
            session_input,
        )

    async def close(self) -> None:
        self._closed = True
        try:
            await self._run_on_runtime_thread(self._close_sync_all_ranks)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        if self._closed:
            raise OmnidreamsRuntimeError("Session is closed.")
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")

        async with self._step_lock:
            if self._closed:
                raise OmnidreamsRuntimeError("Session is closed.")
            return await self._run_on_runtime_thread(
                self._generate_chunk_sync_all_ranks,
                segments,
                frame_times,
            )

    async def render_native_hud_frame(
        self, status_message: str | None = None
    ) -> torch.Tensor | None:
        """Render one exact native HUD frame for WebRTC presentation."""
        async with self._step_lock:
            return await self._run_on_runtime_thread(
                self._render_native_hud_frame_sync, status_message
            )

    async def handle_native_hud_key(
        self, *, key: str, down: bool
    ) -> torch.Tensor | None:
        """Forward a browser key through the shared native presenter."""
        async with self._step_lock:
            return await self._run_on_runtime_thread(
                self._handle_native_hud_key_sync, key, down
            )

    async def handle_native_hud_pointer(
        self, *, x: float, y: float, pressed: bool
    ) -> torch.Tensor | None:
        """Forward a normalized browser click through native HUD hit-testing."""
        async with self._step_lock:
            return await self._run_on_runtime_thread(
                self._handle_native_hud_pointer_sync, x, y, pressed
            )

    def _native_hud_frame_chunk(self) -> torch.Tensor | None:
        if not self._native_hud_frames:
            return None
        frame = np.ascontiguousarray(self._native_hud_frames[-1])
        self._native_hud_frames.clear()
        return (
            torch.from_numpy(frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .contiguous()
        )

    def _render_native_hud_frame_sync(
        self, status_message: str | None
    ) -> torch.Tensor | None:
        hud = self._native_hud
        if hud is None:
            return None
        self._native_hud_frames.clear()
        hud.render_current_frame(status_message)
        return self._native_hud_frame_chunk()

    def _handle_native_hud_key_sync(
        self, key: str, down: bool
    ) -> torch.Tensor | None:
        hud = self._native_hud
        if hud is None:
            return None
        self._native_hud_frames.clear()
        hud.browser_key(key, down)
        keyboard = self._native_hud_keyboard
        if keyboard is not None and keyboard.consume_reset_request():
            self._reset_rollout_sync_all_ranks()
        hud.render_current_frame()
        return self._native_hud_frame_chunk()

    def _handle_native_hud_pointer_sync(
        self, x: float, y: float, pressed: bool
    ) -> torch.Tensor | None:
        hud = self._native_hud
        if hud is None:
            return None
        self._native_hud_frames.clear()
        hud.browser_pointer(x, y, pressed=pressed)
        return self._native_hud_frame_chunk()

    async def _run_on_runtime_thread(
        self,
        func: Callable[..., _T],
        *args: Any,
    ) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._runtime_thread_entry,
            func,
            args,
        )

    def _runtime_thread_entry(
        self,
        func: Callable[..., _T],
        args: tuple[Any, ...],
    ) -> _T:
        device = self._device
        if device is None:
            device = _resolve_cuda_device(self.config.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return func(*args)

    def peek_next_chunk_num_frames(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._state is None:
            return int(self._wrapper.initial_frame_chunk_size)
        return int(self._wrapper.frame_chunk_size)

    def peek_steady_chunk_num_frames(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        return int(self._wrapper.frame_chunk_size)

    @distributed_op(WebRTCControlSignal.INITIALIZE)
    def _initialize_sync_all_ranks(self) -> None:
        self._initialize_sync()

    @distributed_op(WebRTCControlSignal.RESET_SESSION)
    def _reset_rollout_sync_all_ranks(
        self, session_input: OmnidreamsSessionInput | None = None
    ) -> None:
        self._reset_rollout_sync(session_input=session_input)

    @distributed_op(WebRTCControlSignal.ACTION_STEP)
    def _generate_chunk_sync_all_ranks(
        self,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        return self._generate_one_chunk_sync(segments=segments, frame_times=frame_times)

    @distributed_op(WebRTCControlSignal.CLOSE)
    def _close_sync_all_ranks(self) -> None:
        self._close_sync()

    def _initialize_sync(self) -> None:
        if self._wrapper is not None:
            return

        init_t0 = time.perf_counter()
        cfg = self.config
        game_archive_path: Path | None = None
        if cfg.scene_dir is None:
            scene_uuid = cfg.scene_uuid or DEFAULT_WEBRTC_SCENE_UUID
            if cfg.game_mode:
                game_archive_path = hf_hub_download_scene(scene_uuid, cfg.scene_variant)
            scene_dir = _ensure_hf_webrtc_scene_synced(
                scene_uuid,
                variant=cfg.scene_variant,
                prompt_filename=cfg.prompt_filename,
                clipgt_dirname=cfg.clipgt_dirname,
            )
        else:
            scene_source = cfg.scene_dir
            if cfg.game_mode:
                game_archive_path = _resolve_game_scene_archive(
                    scene_source,
                    scene_uuid=cfg.scene_uuid,
                    variant=cfg.scene_variant,
                )
            scene_dir = _extract_local_webrtc_scene_if_needed(
                scene_source,
                scene_uuid=cfg.scene_uuid,
                variant=cfg.scene_variant,
                clipgt_dirname=cfg.clipgt_dirname,
            )

        cfg.scene_dir = scene_dir
        if cfg.server_side_hud:
            self._initialize_native_hud(scene_dir)
        clipgt_dir, first_frame_path, prompt_path = _resolve_webrtc_scene_assets(
            scene_dir,
            prompt_filename=cfg.prompt_filename,
            clipgt_dirname=cfg.clipgt_dirname,
            camera_name=cfg.camera_name,
            variant=cfg.scene_variant,
        )
        if game_archive_path is not None:
            self._game_scene = load_scene_bundle(
                scene_path=game_archive_path,
                camera_name=cfg.camera_name,
                variant=cfg.scene_variant,
                prompt_override=None,
                raster=RasterConfig(width=cfg.video_width, height=cfg.video_height),
            )
        if (
            cfg.pipeline_config is None
            and cfg.pipeline_config_name not in OMNIDREAMS_CONFIGS
        ):
            supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
            raise ValueError(
                f"Unknown pipeline_config_name={cfg.pipeline_config_name!r}. "
                f"Supported: {supported}"
            )
        pipeline_cfg = (
            cfg.pipeline_config or OMNIDREAMS_CONFIGS[cfg.pipeline_config_name]
        )
        transformer_cfg = pipeline_cfg.diffusion_model.transformer
        if not isinstance(transformer_cfg, CosmosTransformerConfig):
            raise TypeError(
                "Omnidreams WebRTC requires a CosmosTransformerConfig pipeline."
            )
        if transformer_cfg.num_views != 1:
            raise ValueError(
                "Omnidreams WebRTC v1 only supports single-view configs; "
                f"{cfg.pipeline_config_name!r} has num_views={transformer_cfg.num_views}."
            )

        self._device = torch.device(cfg.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Omnidreams WebRTC runtime.")

        logger.info("Loading Omnidreams first frame from {}", first_frame_path)
        image_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read first frame from {first_frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (cfg.video_width, cfg.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        self._initial_rgb_frames = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.uint8)
        )

        prompt = prompt_path.read_text(encoding="utf-8").strip() or AV_POSITIVE_PROMPT
        self._text_prompts = [TextPrompt(positive=prompt)]

        loadable_clipgt_dir = self._prepare_clipgt_dir(clipgt_dir)
        logger.info("Loading Omnidreams scene data from {}", loadable_clipgt_dir)
        scene_t0 = time.perf_counter()
        scene_data = load_scene(
            loadable_clipgt_dir,
            camera_names=[cfg.camera_name],
            max_frames=-1,
            input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
            resize_resolution_hw=(cfg.video_height, cfg.video_width),
        )
        logger.info(
            "Loaded Omnidreams scene data in {:.1f}s; attaching Ludus scene.",
            time.perf_counter() - scene_t0,
        )
        ludus_t0 = time.perf_counter()
        scene_data = load_and_attach_ludus_scene(
            loadable_clipgt_dir,
            scene_data,
            device=self._device,
        )
        logger.info(
            "Attached Omnidreams Ludus scene in {:.1f}s.",
            time.perf_counter() - ludus_t0,
        )
        if not scene_data.ego_poses:
            raise ValueError(f"Scene {loadable_clipgt_dir} has no ego poses.")
        if cfg.camera_name not in scene_data.camera_models:
            raise ValueError(
                f"Camera {cfg.camera_name!r} was not loaded from {loadable_clipgt_dir}."
            )
        if cfg.camera_name not in scene_data.camera_extrinsics:
            raise ValueError(
                f"Camera {cfg.camera_name!r} has no extrinsics in {loadable_clipgt_dir}."
            )

        logger.info(
            "Setting up Omnidreams pipeline {} on {}. This may load checkpoints, "
            "compile modules, and initialize CUDA graphs.",
            cfg.pipeline_config_name,
            self._device,
        )
        pipeline_t0 = time.perf_counter()
        self._wrapper = OmnidreamsConditioningWrapper(
            pipeline_config_name=cfg.pipeline_config_name,
            pipeline_config=cfg.pipeline_config,
            resolution_wh=(cfg.video_width, cfg.video_height),
            seed_for_every_rollout=cfg.seed,
            device=self._device,
        )
        logger.info(
            "Omnidreams pipeline setup complete in {:.1f}s.",
            time.perf_counter() - pipeline_t0,
        )
        self._scene_data = scene_data
        logger.info("Creating Omnidreams renderer for camera {}", cfg.camera_name)
        renderer_t0 = time.perf_counter()
        self._renderer = self._wrapper.create_renderer(scene_data, [cfg.camera_name])
        logger.info(
            "Omnidreams renderer ready in {:.1f}s.",
            time.perf_counter() - renderer_t0,
        )
        self._camera_to_rig = torch.as_tensor(
            scene_data.camera_extrinsics[cfg.camera_name],
            device=self._device,
            dtype=torch.float32,
        )
        self._initial_ego_pose = scene_data.ego_poses[0].transformation_matrix
        self._next_timestamp_us = int(scene_data.ego_poses[0].timestamp)
        self._reset_rollout_sync()
        self._initialize_video_encoder_sync()
        logger.info(
            "Omnidreams runtime initialization complete in {:.1f}s.",
            time.perf_counter() - init_t0,
        )

    def _initialize_video_encoder_sync(self) -> None:
        """Select the video encoder for this runtime.

        Runs on the runtime executor thread so any GPU-side probe
        (``CreateEncoder``) sees the same CUDA context the model uses.

        Non-master ranks skip encoder initialization. WebRTC media is
        served only by the master rank, so allocating an NVENC session
        on a worker would consume one of the local GPU's concurrent
        session slots without ever encoding a frame — and could fail
        the worker's startup if the pool cannot accommodate one
        allocation per rank.
        """
        if not self.is_master:
            return
        if self._video_encoder is not None:
            self._video_encoder.close()
            self._video_encoder = None
        device = (
            self._device
            if self._device is not None
            else _resolve_cuda_device(
                self.config.device,
            )
        )
        gpu_id = device.index if device.index is not None else 0
        self._video_encoder = select_encoder(
            backend=self.config.encoder_backend,
            width=self.config.video_width,
            height=self.config.video_height,
            fps=self.config.fps,
            bitrate=self.config.encoder_bitrate_bps,
            gpu_id=gpu_id,
            gop=self.config.encoder_gop,
        )

    def _initialize_native_hud(self, scene_dir: Path) -> None:
        """Build the headless native HUD for WebRTC output."""
        if self._native_hud is not None:
            self._native_hud.close()
        from omnidreams.interactive_drive.demo import _load_control_assets

        cfg = self.config
        scene_option = SimpleNamespace(
            label=scene_dir.name,
            path=scene_dir,
            variants=(cfg.scene_variant,),
            variant_paths={},
            thumbnail=None,
        )
        args = SimpleNamespace(
            scene=scene_dir,
            variant=cfg.scene_variant,
            bev=False,
            bev_resolution="1024x1024",
            bev_height_m=75.0,
            bev_fov_deg=60.0,
            bev_tilt_deg=0.0,
        )
        keyboard = KeyboardState()
        keyboard.set_view_mode("model_rgb")
        self._native_hud_keyboard = keyboard
        self._native_hud = NativeHudBrowserPresenter(
            RasterConfig(width=cfg.video_width, height=cfg.video_height),
            keyboard,
            args=args,
            scene_options=(scene_option,),
            control_assets=_load_control_assets(None),
            frame_sink=self._native_hud_frames.append,
        )
        self._native_hud.set_engine_active(True)
        self._native_hud.set_postprocess_control(
            preset=self.config.postprocess.preset,
            enabled=self.config.postprocess.is_enabled(),
            callback=self._set_postprocess_enabled_sync,
        )

    def _set_postprocess_enabled_sync(self, enabled: bool) -> None:
        """Apply the native HUD post-process toggle to the WebRTC runtime."""
        if enabled:
            self._reset_postprocess_stream(
                OmnidreamsSessionInput(
                    postprocess_preset=self.config.postprocess.preset
                )
            )
        else:
            self._close_postprocess_stream()

    def _prepare_clipgt_dir(self, clipgt_dir: Path) -> Path:
        def _has_prefixed_parquets(path: Path) -> bool:
            return any(path.glob("*.calibration_estimate.parquet"))

        def _has_unprefixed_parquets(path: Path) -> bool:
            return (path / "calibration_estimate.parquet").exists()

        if _has_prefixed_parquets(clipgt_dir):
            return clipgt_dir

        parquet_source_dir: Path | None = None
        if _has_unprefixed_parquets(clipgt_dir):
            parquet_source_dir = clipgt_dir
        else:
            # Some HF scenes extract into ``clipgt/clipgt`` (or another single
            # nested directory) while first_image/prompt stay one level up.
            # Discover that nested parquet root and normalize it for loader use.
            nested_candidates = [
                child for child in clipgt_dir.iterdir() if child.is_dir()
            ]
            for candidate in nested_candidates:
                if _has_prefixed_parquets(candidate):
                    return candidate
                if _has_unprefixed_parquets(candidate):
                    parquet_source_dir = candidate
                    break

        if parquet_source_dir is None:
            return clipgt_dir

        self._clipgt_temp_dir = tempfile.TemporaryDirectory(prefix="omnidreams-clipgt-")
        staged = Path(self._clipgt_temp_dir.name)
        for source in parquet_source_dir.glob("*.parquet"):
            target = staged / f"clip.{source.name}"
            _link_or_copy_file(source.resolve(), target)
        return staged

    def _reset_rollout_sync(
        self, session_input: OmnidreamsSessionInput | None = None
    ) -> None:
        if self._wrapper is None or self._renderer is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._initial_ego_pose is None or self._scene_data is None:
            raise OmnidreamsRuntimeError("Scene state is not initialized.")

        self._reset_postprocess_stream(session_input)
        if self._state is not None and self._state.pipeline_cache is not None:
            del self._state.pipeline_cache
        self._state = None
        if self._game_simulation is not None:
            self._game_simulation.close()
            self._game_simulation = None
        if self.config.game_mode:
            if self._game_scene is None:
                raise OmnidreamsRuntimeError(
                    "Game mode scene physics were not initialized."
                )
            game_scene = self._game_scene
            self._game_simulation = EgoVehicleKinematics(
                initial_state=state_from_initial_pose(
                    initial_rig_to_world=game_scene.initial_rig_to_world,
                    initial_yaw_rad=game_scene.initial_yaw_rad,
                    initial_speed_mps=10.0,
                ),
                vehicle_config=VehicleConfig(
                    actor_collision_enabled=True,
                    static_collision_enabled=True,
                ),
                ground_snapper=build_ground_snapper(game_scene),
                initial_timestamp_us=game_scene.initial_timestamp_us,
                map_bounds=build_map_bounds(game_scene),
                scene=game_scene,
                physics_active_radius_m=self.config.physics_active_radius_m,
            )
        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.pose_integrator.reset(self._initial_ego_pose)
        self.autoregressive_index = 0
        self._next_timestamp_us = int(self._scene_data.ego_poses[0].timestamp)
        self._wrapper.set_rollout_seed(self.config.seed)

    def _close_sync(self) -> None:
        state = self._state
        wrapper = self._wrapper
        self._state = None
        self._wrapper = None
        self._renderer = None
        self._scene_data = None
        self._initial_rgb_frames = None
        self._text_prompts = None
        self._camera_to_rig = None
        self._initial_ego_pose = None
        if self._game_simulation is not None:
            self._game_simulation.close()
            self._game_simulation = None
        self._game_scene = None
        if self._native_hud is not None:
            self._native_hud.close()
            self._native_hud = None
        self._native_hud_keyboard = None
        self._close_postprocess_stream()
        if self._video_encoder is not None:
            self._video_encoder.close()
            self._video_encoder = None

        if state is not None and wrapper is not None:
            wrapper.cleanup(state)
        if wrapper is not None:
            del wrapper
        if self._clipgt_temp_dir is not None:
            self._clipgt_temp_dir.cleanup()
            self._clipgt_temp_dir = None

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _reset_postprocess_stream(
        self, session_input: OmnidreamsSessionInput | None
    ) -> None:
        self._close_postprocess_stream()
        configured = self.config.postprocess
        preset = (
            session_input.postprocess_preset
            if session_input is not None
            and session_input.postprocess_preset is not None
            else configured.preset
        )
        if preset:
            _validate_requested_postprocess_preset(
                requested_preset=preset,
                configured_preset=configured.preset,
            )
        postprocess = VideoPostprocessChainConfig(
            processors=configured.processors,
            preset=preset,
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        postprocess.validate_execution(world_size=world_size)
        self._postprocess_preset = preset
        if not postprocess.is_enabled():
            return
        if not self.is_master and not postprocess.requires_all_ranks(
            world_size=world_size
        ):
            return
        self._postprocess_stream = VideoPostprocessStream(
            postprocess=postprocess,
            output_layout="bvtchw",
            fps=self.config.fps,
            per_view=False,
            world_size=world_size,
        )
        logger.info(
            "Omnidreams WebRTC post-processing enabled with preset {!r}.",
            preset,
        )

    def _close_postprocess_stream(self) -> None:
        if self._postprocess_stream is None:
            return
        self._postprocess_stream.finish()
        self._postprocess_stream = None

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        chunk_started_at = time.perf_counter()
        if (
            self._wrapper is None
            or self._renderer is None
            or self._initial_rgb_frames is None
            or self._text_prompts is None
            or self._camera_to_rig is None
        ):
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._device is None:
            raise OmnidreamsRuntimeError("Runtime device is not initialized.")

        num_frames = self.peek_next_chunk_num_frames()
        if len(frame_times) != num_frames:
            raise OmnidreamsRuntimeError(
                f"Expected {num_frames} frame_times for chunk={self.autoregressive_index}, "
                f"got {len(frame_times)}."
            )
        if not segments:
            raise OmnidreamsRuntimeError(
                f"Chunk={self.autoregressive_index} received empty segments."
            )

        active_keys = segments[-1][2]
        self._sync_native_hud_keys(active_keys)
        physics_started_at = time.perf_counter()
        if self._game_simulation is not None:
            trajectory = self._game_simulation.pose_chunk(
                command=command_from_snapshot(
                    ControlSnapshot(pressed=set(active_keys))
                ),
                chunk_size=num_frames,
                frame_interval_s=1.0 / float(self.config.fps),
                extrapolation_offset_s=0.0,
            )
            ego_poses = trajectory.rig_poses_world
            frame_timestamps_us = [
                int(timestamp) for timestamp in trajectory.timestamps_us
            ]
            if trajectory.actor_collision_detected and self._native_hud is not None:
                self._native_hud.trigger_visual_flare()
        else:
            ego_poses = self.pose_integrator.integrate_chunk(
                segments=segments, frame_times=frame_times
            )
            frame_timestamps_us = self._consume_timestamps(num_frames)
        physics_elapsed_s = time.perf_counter() - physics_started_at
        ego_poses_t = torch.from_numpy(ego_poses).to(
            device=self._device, dtype=torch.float32
        )
        camera_poses = torch.einsum("nij,jk->nik", ego_poses_t, self._camera_to_rig)

        camera_names = [self.config.camera_name]
        camera_poses_per_view = {self.config.camera_name: camera_poses}
        serve_hdmaps = self.config.debug_serve_hdmaps
        model_started_at = time.perf_counter()
        if self._state is None:
            output = self._wrapper.start_generation(
                text_prompts=self._text_prompts,
                initial_rgb_frames=self._initial_rgb_frames,
                renderer=self._renderer,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
            self._state = output.state
        else:
            output = self._wrapper.continue_generation(
                state=self._state,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
            self._state = output.state
        model_elapsed_s = time.perf_counter() - model_started_at

        finalize_started_at = time.perf_counter()
        if self._state.pipeline_cache is not None:
            self._wrapper.finalize_block_generation(
                self._state.pipeline_cache,
                output.finalization_state,
            )
        finalize_elapsed_s = time.perf_counter() - finalize_started_at

        condition_chunk = output.condition_frames
        if serve_hdmaps:
            video_chunk = output.condition_frames
        elif output.rgb_frames is None:
            raise OmnidreamsRuntimeError("Omnidreams WebRTC received no RGB frames.")
        else:
            video_chunk = output.rgb_frames

        postprocess_started_at = time.perf_counter()
        if not serve_hdmaps and self._postprocess_stream is not None:
            video_chunk = self._postprocess_stream.process(
                video_chunk,
                autoregressive_index=self.autoregressive_index,
            )
        postprocess_elapsed_s = time.perf_counter() - postprocess_started_at

        hud_started_at = time.perf_counter()
        if self.is_master and self._native_hud is not None:
            video_chunk = self._render_native_hud_chunk(
                video_chunk,
                condition_chunk,
                frame_timestamps_us,
            )
        hud_elapsed_s = time.perf_counter() - hud_started_at
        result = make_webrtc_step_result(
            chunk_index=self.autoregressive_index,
            video_chunk=video_chunk,
            layout="bvtchw",
            stats={
                "model_step_s": model_elapsed_s,
                "physics_s": physics_elapsed_s,
                "finalize_s": finalize_elapsed_s,
                "pixel_post_s": postprocess_elapsed_s,
                "hud_s": hud_elapsed_s,
                "runtime_total_s": time.perf_counter() - chunk_started_at,
            },
            sync_device=self._device,
        )
        self.autoregressive_index += 1
        return result

    def _sync_native_hud_keys(
        self,
        keys: AbstractSet[str],
    ) -> None:
        hud = self._native_hud
        if hud is None:
            return
        active = {str(key).lower() for key in keys}
        for key in self._native_hud_keys - active:
            hud.browser_key(key, False)
        for key in active - self._native_hud_keys:
            hud.browser_key(key, True)
        self._native_hud_keys = active

    def _render_native_hud_chunk(
        self,
        video_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        timestamps_us: list[int],
    ) -> torch.Tensor:
        hud = self._native_hud
        assert hud is not None
        value_range = "uint8" if video_chunk.dtype == torch.uint8 else "minus_one_one"
        frames = rgb_array_to_uint8_frames(
            video_chunk,
            layout="bvtchw",
            value_range=value_range,
        )
        condition_value_range = (
            "uint8" if condition_chunk.dtype == torch.uint8 else "minus_one_one"
        )
        condition_frames = rgb_array_to_uint8_frames(
            condition_chunk,
            layout="bvtchw",
            value_range=condition_value_range,
        )
        keyboard = self._native_hud_keyboard
        view_mode = keyboard.view_mode if keyboard is not None else "model_rgb"
        self._native_hud_frames.clear()
        for timestamp_us, rgb, condition_rgb in zip(
            timestamps_us, frames, condition_frames, strict=True
        ):
            frame = PresentedFrame(
                timestamp_us=timestamp_us,
                rgb_host_uint8=condition_rgb,
                depth_host_f32=None,
                model_rgb_host_uint8=rgb,
            )
            hud.present_frame(frame, view_mode)
        rendered = np.stack(self._native_hud_frames, axis=0)
        tensor = torch.from_numpy(rendered)
        return tensor.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0).contiguous()

    def _consume_timestamps(self, num_frames: int) -> list[int]:
        step_us = int(round(1_000_000 / self.config.fps))
        timestamps = [self._next_timestamp_us + i * step_us for i in range(num_frames)]
        self._next_timestamp_us += num_frames * step_us
        return timestamps


_ManagedOmnidreamsSession = ManagedWebRTCSession


class OmnidreamsWebRTCSessionManager(
    BaseWebRTCSessionManager[OmnidreamsInferenceRuntime, OmnidreamsRuntimeConfig]
):
    """Owns one active WebRTC session and forwards WSAD actions."""

    _busy_message = "An Omnidreams session is already active."
    _warmup_label = "Omnidreams WebRTC"
    _runtime_error_types = (OmnidreamsRuntimeError,)
    # A chunk-generation failure here is fatal to the rollout, so tear the
    # session down instead of retrying on the next tick.
    _close_session_on_generation_error = True
    _resampler_supported_keys = DRIVING_SUPPORTED_KEYS

    def __init__(
        self,
        *,
        runtime_config: OmnidreamsRuntimeConfig | None = None,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        runtime_config = runtime_config or OmnidreamsRuntimeConfig()
        super().__init__(
            runtime=OmnidreamsInferenceRuntime(config=runtime_config),
            runtime_config=runtime_config,
            fps=runtime_config.fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )
        self._pending_session_input: OmnidreamsSessionInput | None = None

    async def create_answer(
        self, *, offer_sdp: str, offer_type: str
    ) -> dict[str, str]:
        """Create an answer and queue the exact native waiting frame first."""
        answer = await super().create_answer(
            offer_sdp=offer_sdp, offer_type=offer_type
        )
        managed_session = self._active_session
        if managed_session is not None and self.runtime_config.server_side_hud:
            frame = await self._runtime.render_native_hud_frame("Loading Scene...")
            if frame is not None:
                await managed_session.video_track.enqueue_chunk(frame)
        return answer

    async def _handle_event_message(
        self,
        *,
        managed_session: ManagedWebRTCSession,
        payload: dict[str, Any],
    ) -> bool:
        event_id = str(payload.get("event_id", payload.get("id", ""))).strip()
        if event_id == "native_hud_key":
            key = str(payload.get("key", "")).strip()
            if not key:
                return False
            state = str(payload.get("state", "press")).strip().lower()
            frame = await self._runtime.handle_native_hud_key(
                key=key, down=state not in {"clear", "release", "up", "off"}
            )
        elif event_id == "native_hud_pointer":
            try:
                x = float(payload["x"])
                y = float(payload["y"])
            except (KeyError, TypeError, ValueError):
                return False
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                return False
            frame = await self._runtime.handle_native_hud_pointer(
                x=x, y=y, pressed=bool(payload.get("pressed", True))
            )
        else:
            return await super()._handle_event_message(
                managed_session=managed_session, payload=payload
            )

        if frame is not None:
            await managed_session.video_track.enqueue_chunk(frame)
        return True

    def _model_name(self) -> str:
        return self.runtime_config.pipeline_config_name

    def _chunk_done_extra(self) -> dict[str, Any]:
        return {
            "stream": "hdmap" if self.runtime_config.debug_serve_hdmaps else "rgb",
            "postprocess_preset": self._runtime.postprocess_preset,
        }

    def _peek_pending_session_input(self) -> OmnidreamsSessionInput | None:
        return self._pending_session_input

    def _clear_pending_session_input(self) -> None:
        self._pending_session_input = None

    async def _reset_runtime_for_session(
        self, session_input: OmnidreamsSessionInput | None
    ) -> None:
        await self._runtime.reset_for_new_session(session_input=session_input)

    def set_pending_session_input(self, session_input: OmnidreamsSessionInput) -> None:
        if self.has_active_session():
            raise SessionBusyError(self._busy_message)
        preset = session_input.postprocess_preset
        if preset:
            _validate_requested_postprocess_preset(
                requested_preset=preset,
                configured_preset=self.runtime_config.postprocess.preset,
            )
        self._pending_session_input = session_input

    def _register_extra_peer_handlers(self, peer_connection: Any) -> None:
        @peer_connection.on("iceconnectionstatechange")
        def on_iceconnectionstatechange() -> None:
            logger.info(
                "Peer ICE connection state changed: {}",
                peer_connection.iceConnectionState,
            )

        @peer_connection.on("icegatheringstatechange")
        def on_icegatheringstatechange() -> None:
            logger.debug(
                "Peer ICE gathering state changed: {}",
                peer_connection.iceGatheringState,
            )

    def _on_offer_received(self, offer_sdp: str) -> None:
        logger.info(
            "Received WebRTC offer with {}.", _summarize_sdp_candidates(offer_sdp)
        )

    def _on_answer_created(self, answer_sdp: str) -> None:
        logger.info(
            "Created WebRTC answer with {}.", _summarize_sdp_candidates(answer_sdp)
        )
