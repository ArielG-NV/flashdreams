# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import io
import math
import os
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omnidreams import scenes as _scenes
from omnidreams.interactive_drive import cli as _cli
from omnidreams.interactive_drive.app import InteractiveDriveApp
from omnidreams.interactive_drive.config import BevConfig, RasterConfig
from omnidreams.interactive_drive.input.wheel_profiles import (
    WheelProfile,
    create_input_bridge,
    default_controller_profile,
    load_wheel_profiles,
    user_wheel_profiles_dir,
)
from omnidreams.interactive_drive.log import configure_logging
from omnidreams.interactive_drive.synthetic_scene import build_synthetic_scene_to_temp
from omnidreams.scenes import normalise_scene_uuid, scenes_cache_root
from PIL import Image

# Right-side HUD panel width (wheel, pedals, speed, BEV minimap); camera fills
# the rest. Pinned at 500 px because the panel content is asset-driven.
HUD_PANEL_WIDTH = 500

# Bundled AlpaSim-style steering-wheel / pedal PNGs that drive the HUD
# chrome. Resolved relative to the installed package (like the other
# ``cli.py`` defaults) so the realistic controls render out of the box
# regardless of the user's cwd; ``--control-assets-dir`` overrides it.
_BUNDLED_CONTROL_ASSETS_DIR = _cli._PACKAGE_ROOT / "assets" / "wheel_and_pedals"
SCENE_THUMB_SIZE = (140, 64)
KEYBOARD_STEER_SCALE = 0.75
KEYBOARD_STEER_RATE_PER_S = 0.6
KEYBOARD_STEER_RETURN_RATE_PER_S = 1.4
# BEV minimap panel sits at the bottom of the right HUD column.
# Geometry is hand-tuned to leave ~12px gaps to the pedals/edges and
# keeps roughly square aspect to match the BEV camera output.
BEV_PANEL_TOP_GAP = 12
BEV_PANEL_SIDE_MARGIN = 14
BEV_PANEL_BOTTOM_MARGIN = 12
BEV_PANEL_MIN_HEIGHT = 100

# Google-Maps day-mode palette for the BEV filter (:func:`_apply_googlemaps_filter`).
# Warm cream "land" that unrendered/black BEV regions blend toward.
GMAPS_LAND_RGB = (234, 226, 209)
# Off-white "road" tint so lane paint reads as roads, not neon on the cream.
GMAPS_ROAD_RGB = (252, 250, 244)
# Low-contrast warm grey for magenta road boundaries: keeps the edge readable
# while dropping the cream-vs-magenta lightness jump that drove diagonal aliasing.
GMAPS_BOUNDARY_GREY_RGB = (170, 165, 155)
# Pre-built float32 vectors so the per-BEV-frame numpy expression doesn't
# re-allocate these constants each call.
_GMAPS_LAND_FLOAT = np.array(GMAPS_LAND_RGB, dtype=np.float32)
_GMAPS_BOUNDARY_GREY_FLOAT = np.array(GMAPS_BOUNDARY_GREY_RGB, dtype=np.float32)
_GMAPS_TINTED_MUL = (
    0.55 + 0.45 * np.array(GMAPS_ROAD_RGB, dtype=np.float32) / 255.0
).astype(np.float32)

# BEV camera defaults from the canonical :class:`BevConfig` so the HUD's
# ego-marker placement tracks the rasterizer default (tilt=0 centres the
# marker; positive tilt pushes it lower as the camera sees more ahead).
_BEV_DEFAULTS = BevConfig()
BEV_FOV_DEG = _BEV_DEFAULTS.fov_deg
BEV_TILT_DEG = _BEV_DEFAULTS.tilt_deg


@dataclass(frozen=True)
class SceneOption:
    label: str
    path: Path
    variants: tuple[str, ...]
    thumbnail: Image.Image | None = None
    # Per-variant preview thumbnails keyed by variant slug, for the variant
    # dropdown. Variants without a dedicated preview map to the default image
    # so every row still shows a preview.
    variant_thumbnails: dict[str, Image.Image] = field(default_factory=dict)
    # Variant slug -> its USDZ archive. Distinct sibling files for the current
    # per-weather dataset; the single ``path`` for legacy in-zip-variant scenes.
    variant_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class WheelState:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed_mps: float = 0.0
    connected: bool = False
    reverse: bool = False


class KeyboardDriveState:
    def __init__(self, control: Any) -> None:
        # ``control`` is a drive sink with ``set_drive(steer, throttle, brake)``
        # (the HUD's ``KeyboardStateDriveSink``, writing into ``KeyboardState``).
        self._control = control
        self._pressed: set[str] = set()
        self._state = WheelState()
        self._last_update_s = time.monotonic()

    @property
    def state(self) -> WheelState:
        return WheelState(**self._state.__dict__)

    def set_key(self, keysym: str, down: bool) -> bool:
        key = _keyboard_drive_key(keysym)
        if key is None:
            return False
        if down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)
        return True

    def update(self) -> WheelState:
        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self._last_update_s))
        self._last_update_s = now

        target_steer = 0.0
        if {"a", "left"} & self._pressed:
            target_steer += KEYBOARD_STEER_SCALE
        if {"d", "right"} & self._pressed:
            target_steer -= KEYBOARD_STEER_SCALE
        rate = (
            KEYBOARD_STEER_RATE_PER_S
            if abs(target_steer) > 0
            else KEYBOARD_STEER_RETURN_RATE_PER_S
        )
        steer = _move_towards(self._state.steering, target_steer, rate * dt)
        throttle = 1.0 if {"w", "up"} & self._pressed else 0.0
        brake = 1.0 if {"s", "down", "space"} & self._pressed else 0.0
        target_speed = self._update_target_speed(throttle=throttle, brake=brake, dt=dt)
        self._state = WheelState(
            steering=steer,
            throttle=throttle,
            brake=brake,
            target_speed_mps=target_speed,
            connected=False,
        )
        self._control.set_drive(steer=steer, throttle=throttle, brake=brake)
        return self.state

    def clear(self) -> None:
        self._pressed.clear()
        self._state = WheelState()
        self._control.set_drive(steer=0.0, throttle=0.0, brake=0.0)

    def _update_target_speed(
        self, *, throttle: float, brake: float, dt: float
    ) -> float:
        speed = self._state.target_speed_mps
        if throttle > 0.01 and brake <= 0.05:
            accel = 2.0 * throttle * dt
            current = abs(speed)
            high_speed_knee = 22.35
            if current < high_speed_knee:
                taper = max(0.2, 1.0 - (current / high_speed_knee) ** 2 * 0.5)
            else:
                excess = (current - high_speed_knee) / max(1e-6, 36.0 - high_speed_knee)
                taper = max(0.05, 0.5 * (1.0 - excess) ** 3)
            speed += accel * taper
        elif brake > 0.01:
            speed = max(0.0, speed - 12.0 * brake * dt)
        else:
            creep_target = 4.47
            if speed < creep_target + 0.1:
                speed += (creep_target - speed) * 0.18 * dt
            else:
                speed = max(0.0, speed - 0.5 * dt)
        return max(0.0, min(36.0, speed))


@dataclass(frozen=True)
class ControlAssets:
    steering_wheel: Image.Image | None
    throttle_pressed: Image.Image | None
    throttle_unpressed: Image.Image | None
    brake_pressed: Image.Image | None
    brake_unpressed: Image.Image | None

    @property
    def complete(self) -> bool:
        return (
            self.steering_wheel is not None
            and self.throttle_pressed is not None
            and self.throttle_unpressed is not None
            and self.brake_pressed is not None
            and self.brake_unpressed is not None
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the unified ``interactive-drive`` parser.

    Union of: the backend args from
    :func:`omnidreams.interactive_drive.cli.build_parser`; HUD args
    (``--scene-dir``, ``--controller-*``, ...) ignored under ``--no-hud`` /
    ``--stream-mjpeg``; and the ``--no-hud`` toggle (bare Vulkan window).
    """
    parser = _cli.build_parser()
    # Demo-friendly defaults: most users want the world model and the
    # bundled example manifest. The bare cli still defaults to
    # ``raster`` / ``manifest=None`` for unit-test friendliness.
    # Manifest path is rooted at the sample's own packaged ``configs/`` so
    # the default lands on the bundled YAML regardless of the user's cwd
    # (flashdreams workspaces run from the repo root, not the sample dir).
    parser.set_defaults(
        backend="omnidreams",
        manifest=_cli._PACKAGE_ROOT / "configs/example_world_model.yaml",
    )
    parser.description = (
        "Interactive driving demo. Default mode opens a slangpy HUD with"
        " scene/variant selector, BEV minimap, and steering / pedal"
        " overlays, all rendered into a single Vulkan swapchain. Pass"
        " --no-hud to drop the chrome and just open the bare slangpy"
        " Vulkan window, or --stream-mjpeg HOST:PORT to skip the local"
        " window entirely and serve frames to a browser as an MJPEG"
        " HTTP stream (useful on compute-only hosts without a Vulkan"
        " GPU). For a richer browser viewer use the separate"
        " ``omnidreams.webrtc.server`` entry point."
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help=(
            "Skip the HUD chrome and run the backend with a bare slangpy"
            " Vulkan window (matching the legacy lightweight demo)."
        ),
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=scenes_cache_root(),
        help=(
            "Directory of USDZ scenes shown in the HUD scene selector. "
            "Defaults to ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/``, "
            "the shared cache root used by both this demo and the "
            "``omnidreams.webrtc.server`` scene pipeline."
        ),
    )
    parser.add_argument(
        "--auto-start",
        dest="auto_start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Start loading --scene immediately instead of opening the HUD on"
            " Load Scene. Distinct from --preload-scenes (which only warms the"
            " parse cache in the background)."
        ),
    )
    parser.add_argument(
        # Deprecated alias for --auto-start; kept so existing scripts/docs
        # don't break. The old name was easily confused with --preload-scenes.
        "--autoload-scene",
        dest="auto_start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preload-scenes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Parse every scene in --scene-dir in the background at startup so"
            " switching scenes skips the USDZ parse (the per-scene geometry"
            " upload and first-chunk generation still happen on switch)."
            " Off by default; uses more memory the more scenes are staged."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="auto",
        help=(
            "CUDA_VISIBLE_DEVICES for the backend. ``auto`` (default) leaves"
            " whatever the user already exported untouched; a literal value"
            " (e.g. ``0`` or ``1``) is passed through verbatim; empty string"
            " forces the env var unset. The HUD does not auto-pick a GPU --"
            " set CUDA_VISIBLE_DEVICES (or pass an explicit value) on"
            " multi-GPU hosts where the default-zero pick is wrong."
        ),
    )
    parser.add_argument(
        "--controller-profile",
        "--wheel-profile",
        dest="controller_profile",
        default="auto",
        help=(
            "SDL3 controller/wheel mapping name. 'auto' uses the default profile "
            "created by interactive-drive-configuration, then the portable "
            "built-in mapping."
        ),
    )
    parser.add_argument(
        "--controller-profiles-dir",
        "--wheel-profiles-dir",
        dest="controller_profiles_dir",
        type=Path,
        default=_cli._PACKAGE_ROOT / "configs/controllers",
    )
    parser.add_argument(
        "--control-assets-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing AlpaSim-style wheel/pedal PNGs "
            "(steering_wheel.png, throttle_pressed.png, throttle_unpressed.png, "
            "brake_pressed.png / break_pressed.png, brake_unpressed.png / "
            "break_unpressed.png). Defaults to the bundled assets shipped with "
            "the package; pass a directory to override them."
        ),
    )
    parser.add_argument(
        "--no-controller",
        "--no-wheel",
        dest="no_controller",
        action="store_true",
        help="Disable SDL3 controller, wheel, and pedal input.",
    )
    return parser


def _has_discoverable_scenes(scene_dir: Path, scene: Path) -> bool:
    """Whether the scene picker would find any staged USDZ to offer.

    Mirrors :func:`_discover_scene_options`'s directory sweep -- the
    ``--scene-dir`` cache plus the requested scene's own folder -- so the
    default-scene autostage can be skipped when a curated set of scenes is
    already present.
    """
    for directory in (scene_dir, scene.parent):
        resolved = _project_path(directory)
        if resolved.is_dir() and any(resolved.glob("*.usdz")):
            return True
    return False


def _maybe_autostage_scene(scene: Path, *, scene_dir: Path, allow_skip: bool) -> Path:
    """Auto-download the default scene UUID on first launch.

    Triggers only for a missing ``clipgt-<uuid>.usdz`` under the shared scenes
    cache root; external / non-clipgt paths are returned unchanged. With
    ``allow_skip`` (any scene-picker mode), a missing default is skipped when
    the picker already has staged scenes, so a curated set never blocks on the
    default UUID. ``omnidreams-prepare`` remains the way to pre-stage arbitrary
    UUIDs.
    """
    if scene.exists():
        return scene
    if allow_skip and _has_discoverable_scenes(scene_dir, scene):
        logger.info(
            f"[interactive-drive] default scene '{scene.name}' is not staged; "
            f"using the scenes already present under {scene_dir} instead.",
        )
        return scene
    cache_dir = scenes_cache_root().resolve()
    if scene.resolve().parent != cache_dir:
        return scene
    stem = scene.stem
    if not stem.startswith("clipgt-"):
        return scene
    bare_uuid = normalise_scene_uuid(stem)
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            f"Scene '{scene.name}' is not staged yet and HF_TOKEN is not set.\n"
            "Either export HF_TOKEN to enable auto-staging on launch, or run:\n"
            f"  uv run --package flashdreams-omnidreams omnidreams-prepare --scene-uuid {bare_uuid}"
        )
    logger.info(
        f"[interactive-drive] Scene '{stem}' not found locally; "
        "auto-staging from Hugging Face (one-time download)..."
    )
    from omnidreams.prepare import stage_scene

    staged_default = stage_scene(bare_uuid, force=False)
    # Also stage the scene's other weather variants so the HUD shows a
    # Default/Rain/Snow selector; discovery globs the cache dir for them.
    try:
        sibling_variants = [
            variant
            for uuid, variant in _scenes.list_available_scene_files()
            if uuid == bare_uuid and variant != _scenes.SCENE_VARIANT_DEFAULT
        ]
    except Exception as exc:  # noqa: BLE001 - best-effort; base scene already staged
        logger.info(
            f"[interactive-drive] could not enumerate scene variants ({exc}); "
            "staged the base scene only.",
        )
        sibling_variants = []
    for variant in sibling_variants:
        try:
            stage_scene(bare_uuid, variant=variant, force=False)
        except Exception as exc:  # noqa: BLE001 - skip a variant, keep the rest
            logger.info(
                f"[interactive-drive] failed to stage variant {variant!r} "
                f"({exc}); skipping.",
            )
    return staged_default


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    if not args.synthetic_scene:
        # Only the bare ``--no-hud`` backend has no scene picker; the HUD
        # and MJPEG paths both let the user pick from ``--scene-dir``, so a
        # missing default scene there is fine as long as the directory
        # already has other scenes staged (see _maybe_autostage_scene).
        uses_scene_picker = args.stream_mjpeg is not None or not args.no_hud
        args.scene = _maybe_autostage_scene(
            args.scene, scene_dir=args.scene_dir, allow_skip=uses_scene_picker
        )
    # ``--stream-mjpeg`` runs through ``_run_streaming`` so the long-lived
    # MJPEG presenter (HTTP server, browser session) survives across
    # scene-change requests posted by the in-page picker. ``--no-hud``
    # without MJPEG drops straight through to the bare CLI's Vulkan
    # window, which has no scene picker UI of its own. The default path
    # is the slangpy HUD with full chrome.
    if args.stream_mjpeg is not None:
        _run_streaming(args)
        return
    if args.no_hud:
        _cli.run(args)
        return

    _run_slangpy_hud(args)


def _run_slangpy_hud(args: argparse.Namespace) -> None:
    """Run the engine with the slangpy + PIL HUD presenter in one process.

    Builds one ``SlangPyHudPresenter`` and one long-lived
    :class:`InteractiveDriveApp` at startup (model warmup overlaps the
    scene-selection wait), then loops over scene-change requests calling
    ``app.load_scene`` / ``app.run_scene`` per scene. The warmed model and the
    window stay alive across switches (``close_presenter_on_exit=False``); the
    SDL3 controller input binds once to the app's single ``KeyboardState``.
    """
    from omnidreams.interactive_drive.input.keyboard import KeyboardState
    from omnidreams.interactive_drive.slangpy_hud_presenter import (
        KeyboardStateDriveSink,
        SlangPyHudPresenter,
    )

    _apply_cuda_visible_devices_inplace(args.cuda_visible_devices)
    _resolve_demo_paths(args)
    _materialize_synthetic_scene_for_picker(args)
    scene_options = _discover_scene_options(args.scene_dir, args.scene)
    if not args.scene.exists() and scene_options:
        args.scene = scene_options[0].path
    # Validate paths up front so a typo in ``--manifest`` /
    # ``--scene-dir`` / ``--control-assets-dir`` fails immediately,
    # before we open the slangpy window and the user wastes 30s on
    # world-model warmup that's about to ENOENT. Scene path is
    # validated lazily because ``_discover_scene_options`` already
    # backfills ``args.scene`` from the directory, so a missing
    # ``--scene`` is only fatal if the directory is empty too.
    if args.backend == "omnidreams":
        if args.manifest is None:
            raise SystemExit("--manifest is required for the omnidreams backend")
        if not args.manifest.exists():
            raise SystemExit(
                f"--manifest path does not exist: {args.manifest}"
                " (typo? expected a path or bundled config name like "
                "example_world_model.yaml)"
            )
    if not scene_options and not args.scene.exists():
        raise SystemExit(
            f"--scene path does not exist and --scene-dir contains no scenes: {args.scene}"
        )
    control_assets = _load_control_assets(args.control_assets_dir)
    controller_profile = (
        None if args.no_controller else _select_controller_profile(args)
    )

    # Construct the presenter UPFRONT, before any backend, so the demo
    # can open the HUD window in "Load Scene" mode and wait for the
    # user to pick a scene from the dropdown when ``--auto-start``
    # is off. The placeholder ``KeyboardState`` is rebound to each
    # successive ``InteractiveDriveApp``'s real keyboard via
    # ``presenter.bind_keyboard`` in the factory below; no engine is
    # listening to the placeholder, so events are harmlessly dropped
    # during the initial wait.
    placeholder_keyboard = KeyboardState()
    presenter = SlangPyHudPresenter(
        raster=RasterConfig(),
        keyboard=placeholder_keyboard,
        args=args,
        scene_options=scene_options,
        control_assets=control_assets,
        wheel=None,
    )

    # Build the backend + engine ONCE, up front. Constructing the app
    # starts the (scene-independent) model warmup on the pipeline worker
    # thread immediately, so the long weight-load + compile overlaps with
    # the user's scene-selection wait below instead of starting only after
    # the first pick. The app owns one long-lived KeyboardState and rebinds
    # the presenter to it; scenes are switched in place via
    # ``app.load_scene`` so the warmed model is never rebuilt.
    config, backend = _cli.prepare_config_and_backend(args)
    app = InteractiveDriveApp(
        config=config,
        backend=backend,
        presenter=presenter,
        close_presenter_on_exit=False,
    )
    presenter.set_model_status(can_prewarm=app.can_prewarm, ready_probe=app.model_ready)
    presenter.set_postprocess_control(
        preset=config.postprocess.preset,
        enabled=config.postprocess.is_enabled(),
        callback=app.set_postprocess_enabled,
    )

    # Attach SDL3 input to the presenter's existing event loop. Gamepads use
    # semantic callbacks; wheels and pedal sets use SDL's portable joystick poll.
    wheel: Any = None
    if controller_profile is not None:
        wheel = create_input_bridge(
            profile=controller_profile,
            control=KeyboardStateDriveSink(app.keyboard),
        )
        presenter.set_wheel(wheel)
        wheel.start()

    if args.preload_scenes:
        app.preload_scenes(
            (opt.path, variant, args.prompt)
            for opt in scene_options
            for variant in (opt.variants or ("default",))
        )
        # Lock scene selection until every scene is cached so the user only
        # ever hits the instant (cache-hit) switch path.
        presenter.set_scene_selection_locked(app.preload_in_progress)

    # First scene: prefer the resolved ``config.scene_path`` so
    # ``--synthetic-scene`` (materialised to a temp USDZ) and any autostaged
    # default are honoured; a dropdown selection overrides it below.
    scene_path: Any = config.scene_path
    variant = _resolve_scene_variant(scene_options, scene_path, config.variant)
    presenter.acknowledge_scene_change(scene_path, variant)
    try:
        # ``need_selection`` drives the scene-selection wait: True on first
        # launch (unless ``--auto-start``) and again every time the user
        # exits a scene back to the selector. While waiting the engine is
        # idle, so the video model stops generating -- the whole point of the
        # exit-scene affordance for long-running demos -- without closing the
        # window or dropping the warmed model.
        need_selection = not args.auto_start
        # --auto-start + --preload-scenes: wait for the preloader to finish
        # before the auto-load below so it hits the cache instead of racing
        # the background thread with a second parse of the same USDZ.
        if args.auto_start and app.preload_in_progress():
            presenter.wait_while_preloading(app.preload_in_progress)
        while True:
            if need_selection:
                request = presenter.wait_for_scene_selection()
                if request is None:
                    break  # window closed before any scene was loaded
                scene_path, variant = request
                presenter.acknowledge_scene_change(scene_path, variant)
                need_selection = False

            presenter.set_engine_active(True)
            # load_scene parses the USDZ on a background thread while keeping
            # the window responsive; it returns False if the window closed
            # (or a new scene was requested) before the parse finished, so
            # we skip run_scene and let the pending checks below decide
            # whether to exit the scene, switch scenes, or quit.
            if app.load_scene(scene_path, variant, args.prompt):
                app.run_scene()
            presenter.set_engine_active(False)
            if presenter.pending_exit_scene:
                # ``x`` / bound exit button: tear down the rollout and go
                # back to the selector over the same presenter.
                presenter.acknowledge_exit_scene()
                need_selection = True
                continue
            requested = presenter.pending_scene_change
            if requested is None:
                # Window closed (X / ESC) during load or run; we're done.
                break
            scene_path, variant = requested
            presenter.acknowledge_scene_change(scene_path, variant)
    finally:
        app.shutdown()
        presenter.close()


def _run_streaming(args: argparse.Namespace) -> None:
    """Run the engine with the MJPEG streaming presenter and a scene-change loop.

    Like :func:`_run_slangpy_hud` but with a long-lived
    :class:`MJPEGStreamingPresenter`: the HTTP server / browser sessions stay
    alive across scene swaps while only the scene is rebuilt. Scene options are
    serialised to JSON for the in-browser ``/scenes`` dropdown.
    """
    from omnidreams.interactive_drive.input.keyboard import KeyboardState
    from omnidreams.interactive_drive.streaming_presenter import (
        MJPEGStreamingPresenter,
        parse_bind,
    )

    _apply_cuda_visible_devices_inplace(args.cuda_visible_devices)
    _resolve_demo_paths(args)
    _materialize_synthetic_scene_for_picker(args)
    scene_options = _discover_scene_options(args.scene_dir, args.scene)
    if not args.scene.exists() and scene_options:
        args.scene = scene_options[0].path
    if args.backend == "omnidreams":
        if args.manifest is None:
            raise SystemExit("--manifest is required for the omnidreams backend")
        if not args.manifest.exists():
            raise SystemExit(
                f"--manifest path does not exist: {args.manifest}"
                " (typo? expected a path or bundled config name like "
                "example_world_model.yaml)"
            )
    if not scene_options and not args.scene.exists():
        raise SystemExit(
            f"--scene path does not exist and --scene-dir contains no scenes: {args.scene}"
        )

    # JSON-serialisable form of the discovered scenes for the browser
    # ``/scenes`` endpoint. Thumbnails are JPEG-encoded once at startup
    # and stashed on the presenter so the per-card ``/thumbnail``
    # request just blobs the bytes back -- no per-request encode cost
    # under the HTTP handler thread, which would otherwise compete
    # with the main camera's encode budget.
    scenes_payload: tuple[dict[str, object], ...] = tuple(
        {
            "label": opt.label,
            "path": str(opt.path),
            "variants": list(opt.variants),
        }
        for opt in scene_options
    )
    thumbnails: dict[str, bytes] = {}
    for opt in scene_options:
        if opt.thumbnail is None:
            continue
        buf = io.BytesIO()
        # PIL's RGBA / palette-mode thumbnails need an explicit RGB
        # conversion before JPEG encode. The discovery layer already
        # returns RGB, but be defensive in case it changes upstream.
        thumb_rgb = (
            opt.thumbnail
            if opt.thumbnail.mode == "RGB"
            else opt.thumbnail.convert("RGB")
        )
        thumb_rgb.save(buf, format="JPEG", quality=85)
        thumbnails[str(opt.path)] = buf.getvalue()

    bind_host, bind_port = parse_bind(args.stream_mjpeg)
    placeholder_keyboard = KeyboardState()
    presenter = MJPEGStreamingPresenter(
        raster=RasterConfig(),
        keyboard=placeholder_keyboard,
        bind_host=bind_host,
        bind_port=bind_port,
        scenes=scenes_payload,
        thumbnails=thumbnails,
    )

    # Build the backend + engine once so the model warms up (on the
    # pipeline worker thread) while the browser is still choosing the first
    # scene. The app rebinds the presenter to its long-lived keyboard and
    # switches scenes in place via ``app.load_scene``, keeping the warmed
    # model resident across scene changes.
    config, backend = _cli.prepare_config_and_backend(args)
    app = InteractiveDriveApp(
        config=config,
        backend=backend,
        presenter=presenter,
        close_presenter_on_exit=False,
    )
    presenter.set_model_status(can_prewarm=app.can_prewarm, ready_probe=app.model_ready)

    if args.preload_scenes:
        app.preload_scenes(
            (opt.path, variant, args.prompt)
            for opt in scene_options
            for variant in (opt.variants or ("default",))
        )
        # Lock scene selection until every scene is cached so the user only
        # ever hits the instant (cache-hit) switch path.
        presenter.set_scene_selection_locked(app.preload_in_progress)

    try:
        if args.auto_start:
            # Headless / scriptable start: skip the browser scene picker and
            # load the resolved ``--scene`` (or the first discovered scene)
            # immediately. This lets the demo run with no GUI/browser.
            # --auto-start + --preload-scenes: let the preloader finish first
            # so the auto-load hits the cache instead of racing a second parse.
            if app.preload_in_progress():
                presenter.wait_while_preloading(app.preload_in_progress)
            scene_path = config.scene_path
            variant = _resolve_scene_variant(scene_options, scene_path, config.variant)
            presenter.acknowledge_scene_change(scene_path, variant)
            logger.info(
                f"[demo] streaming auto-start scene -> {scene_path.name} "
                f"variant={variant!r}",
            )
        else:
            # Don't auto-load: always wait for the browser to pick the first
            # scene. There's no Vulkan window to show progress in, so the
            # presenter publishes an idle overlay frame ("Loading world
            # model..." while warmup runs in the background, then "Select a
            # scene to begin") so connected browsers have something to render
            # while the wait spins.
            logger.info(
                "[demo] streaming presenter waiting for first scene selection...",
            )
            request = presenter.wait_for_scene_selection()
            if request is None:
                return  # presenter closed before any selection (Ctrl-C)
            scene_path, variant = request
            presenter.acknowledge_scene_change(scene_path, variant)
            logger.info(
                f"[demo] streaming initial scene -> {scene_path.name} variant={variant!r}",
            )

        while True:
            # load_scene parses the USDZ on a background thread while the
            # browser keeps receiving frames; False means the session is
            # ending (or a new scene was requested) before the parse
            # finished, so skip run_scene and let the check below decide.
            if app.load_scene(scene_path, variant, args.prompt):
                app.run_scene()
            requested = presenter.pending_scene_change
            if requested is None:
                # Either the process is shutting down (Ctrl-C) or the
                # rollout finished without a scene-change request.
                # ``MJPEGStreamingPresenter`` has no native quit
                # affordance, so a "no pending change" exit is
                # treated as the end of the session.
                break
            scene_path, variant = requested
            presenter.acknowledge_scene_change(scene_path, variant)
            logger.info(
                f"[demo] streaming scene change -> {scene_path.name} "
                f"variant={variant!r}",
            )
    finally:
        app.shutdown()
        presenter.close()


def _apply_cuda_visible_devices_inplace(requested: str) -> None:
    """Resolve ``--cuda-visible-devices`` into ``os.environ`` before backend build.

    Must run before ``_cli.run`` (which imports torch.cuda). ``auto`` leaves
    the user's existing export untouched (no auto GPU pick); ``""`` unsets it;
    any other value is passed through verbatim.
    """
    if requested == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    if requested != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = requested


def _resolve_demo_paths(args: argparse.Namespace) -> None:
    for attr in ("scene", "scene_dir", "controller_profiles_dir"):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, _project_path(value))
    if args.manifest is not None:
        args.manifest = _cli.resolve_manifest_path(args.manifest)
    if args.control_assets_dir is not None:
        args.control_assets_dir = _project_path(args.control_assets_dir)


def _materialize_synthetic_scene_for_picker(args: argparse.Namespace) -> None:
    """Build ``--synthetic-scene`` before scene-picker discovery.

    The single-scene ``--no-hud`` path lets ``cli.prepare_config_and_backend``
    materialize the synthetic USDZ. HUD and MJPEG modes discover scenes first
    so the picker can show options before a scene is loaded; those modes need
    the temporary USDZ to exist before discovery runs.
    """
    if not args.synthetic_scene:
        return
    scene_path = build_synthetic_scene_to_temp(
        initial_rgb_path=args.synthetic_initial_rgb,
        prompt=args.synthetic_prompt,
    )
    logger.info(
        "[interactive-drive] synthetic scene materialised at {}",
        scene_path,
    )
    args.scene = scene_path
    # The synthetic inputs have been consumed into the temp USDZ. Clear them so
    # the later shared backend builder treats the scene as a normal archive.
    args.synthetic_scene = False
    args.synthetic_initial_rgb = None
    args.synthetic_prompt = None


def _project_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    # Resolve relative paths against the cwd (standard CLI convention, what
    # users expect when running from the repo root).
    return (Path.cwd() / path).resolve()


def _discover_scene_options(
    scene_dir: Path, selected_scene: Path
) -> tuple[SceneOption, ...]:
    paths: set[Path] = set()
    if selected_scene.exists():
        paths.add(selected_scene.resolve())
    if scene_dir.is_dir():
        paths.update(path.resolve() for path in scene_dir.glob("*.usdz"))
    if selected_scene.parent.is_dir():
        paths.update(path.resolve() for path in selected_scene.parent.glob("*.usdz"))

    # Group archives by scene UUID so the per-weather sibling files
    # (``clipgt-<uuid>-<variant>.usdz``) collapse into one scene with a variant
    # selector. Single-archive scenes stay a group of one.
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(paths):
        uuid, variant = _scenes.parse_scene_stem(path.stem)
        grouped.setdefault(uuid, {})[variant] = path

    options = tuple(
        _scene_option_for_group(variant_paths)
        for _uuid, variant_paths in sorted(grouped.items())
    )
    logger.info(
        "[demo] discovered scenes: "
        + (
            ", ".join(
                f"{scene.label} [{', '.join(scene.variants)}]" for scene in options
            )
            if options
            else "<none>"
        ),
    )
    return options


def _order_variants(variants: Iterable[str]) -> tuple[str, ...]:
    """Order variant slugs with ``default`` first, then the rest sorted."""
    unique = set(variants)
    ordered = ["default"] if "default" in unique else []
    ordered.extend(sorted(unique - {"default"}))
    return tuple(ordered)


def _scene_option_for_group(variant_paths: dict[str, Path]) -> SceneOption:
    """Build one :class:`SceneOption` from a scene's variant archive(s).

    Multiple siblings => the weather variants are the files. A single archive
    => fall back to in-zip variant discovery (legacy / synthetic scenes).
    """
    if len(variant_paths) > 1:
        variants = _order_variants(variant_paths.keys())
        base_path = variant_paths.get("default") or variant_paths[variants[0]]
        resolved_paths = dict(variant_paths)
        variant_thumbnails = _load_variant_file_thumbnails(resolved_paths, variants)
    else:
        base_path = next(iter(variant_paths.values()))
        variants = _discover_variants(base_path)
        resolved_paths = {variant: base_path for variant in variants}
        variant_thumbnails = _load_variant_thumbnails(base_path, variants)
    # Use the first variant's preview for the scene row so the scene and
    # variant dropdowns agree, falling back to the standalone loader.
    thumbnail = (
        variant_thumbnails.get(variants[0])
        or variant_thumbnails.get("default")
        or _load_scene_thumbnail(base_path)
    )
    return SceneOption(
        label=_scene_label(base_path),
        path=base_path,
        variants=variants,
        thumbnail=thumbnail,
        variant_thumbnails=variant_thumbnails,
        variant_paths=resolved_paths,
    )


def _scene_label(path: Path) -> str:
    scene_names = {
        "0d404ff7-2b66-498c-b047-1ed8cded60d4": "Quiet Suburban Boulevard",
        "7bd1eb2f-c375-44ee-b4ca-55473e0773a9": "Late Night Arrival in the Neighborhood",
        "e2993759-36e1-4d97-868f-e2a737f1eb68": "Afternoon Commute Past the Park",
    }
    # Key by bare UUID so the label is stable across weather variant archives.
    uuid, _variant = _scenes.parse_scene_stem(path.stem)
    return scene_names.get(uuid, path.stem)


def _discover_variants(scene_path: Path) -> tuple[str, ...]:
    variants: set[str] = set()
    try:
        with zipfile.ZipFile(scene_path, "r") as zf:
            for name in zf.namelist():
                if "/" in name:
                    continue
                stem = Path(name).stem
                if name.startswith("first_image") and name.endswith(".png"):
                    variant = _scenes.variant_from_stem(stem, "first_image")
                elif name.startswith("prompt") and name.endswith(".txt"):
                    variant = _scenes.variant_from_stem(stem, "prompt")
                else:
                    continue
                if variant is not None:
                    variants.add(variant)
    except (OSError, zipfile.BadZipFile):
        return ("default",)
    # A bare ``default`` (prompt.txt / first_image.png) duplicates the first
    # numbered variant, so when numbered variants exist we expose just those --
    # "1" is then the default selection. Scenes with no numbered variants show
    # a single "default".
    numbered = [value for value in variants if value != "default"]
    if numbered:
        numbered.sort(key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else v))
        return tuple(numbered)
    return ("default",)


def _resolve_scene_variant(
    scene_options: tuple[SceneOption, ...], scene_path: Any, variant: str
) -> str:
    """Return a variant that actually exists for *scene_path*.

    Numbered scenes no longer carry a bare ``default`` entry, so a configured
    ``--variant default`` (or anything the scene lacks) falls back to the
    scene's first variant rather than a selection the dropdown can't show.
    For weather sibling archives, the path itself is also a source of truth:
    ``clipgt-...-snow.usdz`` with the default CLI variant should start as
    ``snow``, not silently load the clear/base archive.
    """
    for option in scene_options:
        path_variant = _scene_option_variant_for_path(option, scene_path)
        if path_variant is None:
            continue
        if variant in option.variants:
            if variant == "default" and path_variant != "default":
                return path_variant
            return variant
        if path_variant in option.variants:
            return path_variant
        return option.variants[0] if option.variants else variant
    return variant


def _scene_option_variant_for_path(option: SceneOption, scene_path: Any) -> str | None:
    try:
        resolved = Path(str(scene_path)).resolve()
    except OSError:
        resolved = None
    raw = str(scene_path)

    # ``variant_paths`` is the authoritative map for weather sibling archives.
    # For legacy single-archive scenes it maps every in-zip variant to the same
    # path, so the first variant intentionally matches the old fallback.
    for variant, path in option.variant_paths.items():
        if _same_scene_path(path, raw, resolved):
            return variant
    if _same_scene_path(option.path, raw, resolved):
        if "default" in option.variants:
            return "default"
        return option.variants[0] if option.variants else None
    return None


def _same_scene_path(path: Path, raw: str, resolved: Path | None) -> bool:
    return (resolved is not None and path == resolved) or str(path) == raw


def _load_scene_thumbnail(scene_path: Path) -> Image.Image | None:
    try:
        with zipfile.ZipFile(scene_path, "r") as zf:
            names = [
                name
                for name in zf.namelist()
                if "/" not in name
                and name.startswith("first_image")
                and name.endswith(".png")
            ]
            if not names:
                return None
            name = "first_image.png" if "first_image.png" in names else sorted(names)[0]
            with Image.open(io.BytesIO(zf.read(name))) as image:
                return _make_thumbnail(image.convert("RGB"), SCENE_THUMB_SIZE)
    except (OSError, zipfile.BadZipFile):
        return None


def _load_variant_thumbnails(
    scene_path: Path, variants: tuple[str, ...]
) -> dict[str, Image.Image]:
    """Per-variant preview thumbnails for the HUD variant dropdown.

    Mirrors :func:`scene_loader._discover_first_images`: a bundle may ship
    ``first_image_<variant>.png`` per variant alongside ``first_image.png``
    (the ``"default"`` variant). Each referenced image is decoded once;
    variants without a dedicated image fall back to the default so every
    dropdown row still shows a preview. Returns an empty mapping when the
    archive has no parseable first images.
    """
    decoded: dict[str, Image.Image] = {}
    try:
        with zipfile.ZipFile(scene_path, "r") as zf:
            names_by_variant: dict[str, str] = {}
            for name in zf.namelist():
                if (
                    "/" in name
                    or not name.startswith("first_image")
                    or not name.endswith(".png")
                ):
                    continue
                variant = _scenes.variant_from_stem(Path(name).stem, "first_image")
                if variant is not None:
                    names_by_variant[variant] = name
            for variant, name in names_by_variant.items():
                with Image.open(io.BytesIO(zf.read(name))) as image:
                    decoded[variant] = _make_thumbnail(
                        image.convert("RGB"), SCENE_THUMB_SIZE
                    )
    except (OSError, zipfile.BadZipFile):
        return {}
    if not decoded:
        return {}
    default = decoded.get("default") or next(iter(decoded.values()))
    return {variant: decoded.get(variant, default) for variant in variants}


def _load_variant_file_thumbnails(
    variant_paths: dict[str, Path], variants: tuple[str, ...]
) -> dict[str, Image.Image]:
    """Per-variant thumbnails when each variant is its own archive.

    Each preview comes from that variant file's ``first_image.png``; variants
    with no usable preview reuse the default. Empty mapping if nothing decoded.
    """
    decoded: dict[str, Image.Image] = {}
    for variant in variants:
        path = variant_paths.get(variant)
        if path is None:
            continue
        thumb = _load_scene_thumbnail(path)
        if thumb is not None:
            decoded[variant] = thumb
    if not decoded:
        return {}
    fallback = decoded.get("default") or next(iter(decoded.values()))
    return {variant: decoded.get(variant, fallback) for variant in variants}


def _make_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = Image.new("RGB", size, (20, 20, 30))
    fitted = _fit_image(image, size)
    thumb.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return thumb


def _variant_label(variant: str) -> str:
    labels = {
        # Per-weather variant archives.
        "default": "Default (Clear)",
        "clear": "Clear",
        "snow": "Snowstorm",
        "rain": "Night Rain",
        # Legacy in-archive numbered variants.
        "1": "Bright Midday Sun",
        "2": "Snowstorm",
        "3": "Night with Heavy Rain",
    }
    return labels.get(variant, variant)


def _merged_controller_profiles(cli_profiles_dir: Path) -> tuple[WheelProfile, ...]:
    """Load user mappings before optional packaged mappings."""
    merged: dict[str, WheelProfile] = {}
    for profile in (
        *load_wheel_profiles(user_wheel_profiles_dir()),
        *load_wheel_profiles(cli_profiles_dir),
    ):
        merged.setdefault(profile.name.lower(), profile)
    return tuple(merged.values())


def _select_controller_profile(args: argparse.Namespace) -> WheelProfile:
    """Choose an SDL3 semantic-gamepad or generic-joystick profile."""
    profiles = _merged_controller_profiles(args.controller_profiles_dir)
    requested = str(args.controller_profile)
    if requested.lower() == "auto":
        profile = next((item for item in profiles if item.is_default), None)
        if profile is None:
            profile = profiles[0] if profiles else default_controller_profile()
        logger.info(f"[demo] SDL3 input mapping={profile.name} backend={profile.backend}")
        return profile

    normalized = requested.lower().replace("_", "-")
    for profile in profiles:
        if profile.name.lower().replace("_", "-") == normalized:
            return profile
    available = ", ".join(profile.name for profile in profiles)
    suffix = f", {available}" if available else ""
    raise SystemExit(
        f"Unknown controller profile {requested!r}. Available profiles: auto{suffix}"
    )


def _load_control_assets(control_assets_dir: Path | None) -> ControlAssets:
    assets_dir = control_assets_dir or _BUNDLED_CONTROL_ASSETS_DIR
    if not assets_dir.is_dir():
        if control_assets_dir is not None:
            logger.info(
                f"[demo] control assets not found at {assets_dir}; using vector fallback",
            )
        return ControlAssets(
            steering_wheel=None,
            throttle_pressed=None,
            throttle_unpressed=None,
            brake_pressed=None,
            brake_unpressed=None,
        )

    # Brake PNGs are accepted under either spelling: the AlpaSim asset
    # bundle ships them as ``break_*.png`` (a typo we inherit), but if a
    # downstream user renames them to the correct ``brake_*.png`` we
    # don't want to silently fall back to the vector renderer.
    assets = ControlAssets(
        steering_wheel=_load_asset_image(assets_dir / "steering_wheel.png"),
        throttle_pressed=_load_asset_image(assets_dir / "throttle_pressed.png"),
        throttle_unpressed=_load_asset_image(assets_dir / "throttle_unpressed.png"),
        brake_pressed=_load_first_asset_image(
            assets_dir, ("brake_pressed.png", "break_pressed.png")
        ),
        brake_unpressed=_load_first_asset_image(
            assets_dir, ("brake_unpressed.png", "break_unpressed.png")
        ),
    )
    if assets.complete:
        logger.info(f"[demo] loaded AlpaSim control assets from {assets_dir}")
    else:
        logger.info(
            f"[demo] incomplete control assets at {assets_dir}; missing files use vector fallback",
        )
    return assets


def _load_asset_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            return image.convert("RGBA").copy()
    except OSError:
        return None


def _load_first_asset_image(
    assets_dir: Path, candidate_filenames: tuple[str, ...]
) -> Image.Image | None:
    """Return the first existing asset image among the given filenames.

    Used to accept either spelling of the brake PNG (``brake_*.png`` vs
    the typo'd ``break_*.png`` shipped by AlpaSim).
    """
    for name in candidate_filenames:
        loaded = _load_asset_image(assets_dir / name)
        if loaded is not None:
            return loaded
    return None


def _move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def _apply_googlemaps_filter(rgb_image: Image.Image) -> Image.Image:
    """Restyle a BEV frame as a Google-Maps minimap (single numpy expression).

    Blends dark regions toward cream "land" and rendered features toward an
    off-white "road" tone. The presence curve has a hard low knee so JPEG
    ringing around high-contrast edges collapses to land instead of surviving
    as dirty grey halos.
    """
    # Already RGB-mode, so skip ``convert``; ``np.asarray`` is zero-copy.
    arr = np.asarray(rgb_image, dtype=np.float32)
    # Recolour magenta road boundaries to low-contrast grey for soft
    # Google-Maps-style borders. Loose detection on purpose so anti-aliased
    # edge pixels get caught too, killing the JPEG/MSAA halo.
    is_magenta = (
        (arr[..., 0] > 130)
        & (arr[..., 2] > 130)
        & (arr[..., 1] < arr[..., 0] * 0.55)
        & (arr[..., 1] < arr[..., 2] * 0.55)
    )
    # In-place recolour avoids the ~3 MB allocation that ``np.where``
    # would do every BEV frame at 512x512.
    np.copyto(arr, _GMAPS_BOUNDARY_GREY_FLOAT, where=is_magenta[..., np.newaxis])
    bright = arr.max(axis=2, keepdims=True) / 255.0
    # Tight knee: < 0.14 collapses to land, > 0.21 fully drawn (0.07-wide
    # blend band) so JPEG ringing / resize halos don't survive as grey outlines.
    presence = np.clip((bright - 0.14) / 0.07, 0.0, 1.0)
    # Tint feature pixels toward the road colour while keeping their
    # original chroma so yellow lane paint stays warmer than white paint.
    tinted = arr * _GMAPS_TINTED_MUL
    out = tinted * presence + _GMAPS_LAND_FLOAT * (1.0 - presence)
    return Image.fromarray(out.clip(0.0, 255.0).astype(np.uint8))


def _bev_marker_y_rel() -> float:
    """Where the rig projects in the BEV image, as a fraction of height.

    Pure top-down (``BEV_TILT_DEG == 0``) puts the rig at image centre
    (0.5). Each degree of forward tilt moves it lower, by
    ``focal_y * tan(tilt) / height = tan(tilt) / (2 * tan(fov/2))``,
    which is the standard pinhole projection of a point on the rig
    plane straight below the camera.
    """
    half_fov = math.radians(BEV_FOV_DEG / 2.0)
    if half_fov <= 0:
        return 0.5
    return min(
        0.95, 0.5 + math.tan(math.radians(BEV_TILT_DEG)) / (2.0 * math.tan(half_fov))
    )


def _keyboard_drive_key(keysym: str) -> str | None:
    mapping = {
        "w": "w",
        "W": "w",
        "a": "a",
        "A": "a",
        "s": "s",
        "S": "s",
        "d": "d",
        "D": "d",
        "Up": "up",
        "Down": "down",
        "Left": "left",
        "Right": "right",
        "space": "space",
    }
    return mapping.get(keysym)


def _fit_image(image: Image.Image, bounds_wh: tuple[int, int]) -> Image.Image:
    max_w, max_h = bounds_wh
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    if size == image.size:
        # PIL's ``Image.resize`` runs ``.copy()`` on same-size input; skip it.
        return image
    return image.resize(size, Image.Resampling.BILINEAR)


if __name__ == "__main__":
    main()
