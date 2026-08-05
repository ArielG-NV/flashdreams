# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless browser transport for the native interactive-drive HUD."""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from omnidreams.interactive_drive.config import BevConfig, RasterConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.physx_debug import select_presented_rgb
from omnidreams.interactive_drive.slangpy_hud_presenter import (
    DEFAULT_WINDOW_H,
    DEFAULT_WINDOW_W,
    HUD_PANEL_WIDTH,
    MIN_WINDOW_H,
    SlangPyHudPresenter,
    _allocate_canvas,
    _LRUCache,
    _resolve_font,
)
from omnidreams.interactive_drive.types import PresentedFrame
from omnidreams.interactive_drive.visual_flare import CollisionVisualFlare, darken_rgb


class NativeHudBrowserPresenter(SlangPyHudPresenter):
    """Render the native HUD into host frames without creating a window."""

    def __init__(
        self,
        raster: RasterConfig,
        keyboard: KeyboardState,
        *,
        args: Any,
        scene_options: tuple[Any, ...],
        control_assets: Any,
        frame_sink: Callable[[np.ndarray], None],
        width: int = DEFAULT_WINDOW_W,
        height: int = DEFAULT_WINDOW_H,
    ) -> None:
        self._raster = raster
        self._keyboard = keyboard
        self._visual_flare = CollisionVisualFlare()
        self._args = args
        bev_width, bev_height = (
            int(component)
            for component in str(args.bev_resolution).lower().split("x", maxsplit=1)
        )
        self._bev_config = BevConfig(
            enabled=bool(args.bev),
            width=bev_width,
            height=bev_height,
            height_m=float(args.bev_height_m),
            fov_deg=float(args.bev_fov_deg),
            tilt_deg=float(args.bev_tilt_deg),
        )
        self._scene_options = scene_options
        self._control_assets = control_assets
        self._wheel = None
        self._frame_sink = frame_sink

        from omnidreams.interactive_drive.demo import (
            KeyboardDriveState,
            _scene_label,
        )
        from omnidreams.interactive_drive.slangpy_hud_presenter import (
            KeyboardStateDriveSink,
        )

        self._keyboard_drive = KeyboardDriveState(
            KeyboardStateDriveSink(keyboard, source="keyboard")
        )
        self._scene_label_fn = _scene_label
        self._configured_size = (max(640, int(width)), max(360, int(height)))
        self._cuda_hud_interop = None
        self._retired_cuda_hud_interops: list[Any] = []
        self._cuda_hud_error_logged = False
        self._pending_resize = None
        self._auto_sized_camera_src_size = None

        self._font_tiny = _resolve_font(14)
        self._font_small = _resolve_font(18)
        self._font_medium = _resolve_font(22)
        self._font_large = _resolve_font(44)
        self._font_speed = _resolve_font(76)
        self._panel_chrome_cache_key = None
        self._panel_chrome_cache = None
        self._speed_chip_cache = _LRUCache(maxsize=64)
        self._wheel_base_image = None
        self._wheel_base_size = None
        self._wheel_rotation_cache = _LRUCache(maxsize=480)
        self._pedal_cache = _LRUCache(maxsize=16)
        self._scene_thumb_cache: dict[Any, Any] = {}
        self._variant_thumb_cache: dict[tuple[Any, str], Any] = {}
        self._bev_panel_cache_key = None
        self._bev_panel_cache = None
        self._bev_panel_epoch = 0
        self._bev_panel_exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="interactive-drive-browser-bev",
        )
        self._bev_panel_future = None

        self._latest_camera_pil = None
        self._latest_bev_source = None
        self._latest_ego_dimensions_lwh = None
        self._prepared_bev_source_key = None
        self._bev_source_generation = 0
        self._latest_camera_rgba = None
        self._latest_camera_src_size = None
        self._camera_resize_cache_key = None
        self._camera_resize_cache = None
        self._camera_texture = None
        self._camera_texture_size = None
        self._camera_fit_texture = None
        self._camera_fit_size = None
        self._camera_rgba_staging = None
        self._canvas_buffer, self._canvas = _allocate_canvas(*self._configured_size)

        self._scene_dropdown_open = False
        self._variant_dropdown_open = False
        self._scene_header_rect = None
        self._variant_header_rect = None
        self._postprocess_rect = None
        self._scene_item_rects: list[tuple[tuple[int, int, int, int], Any]] = []
        self._variant_item_rects: list[tuple[tuple[int, int, int, int], str]] = []
        self._hovered_scene_label = None
        self._hovered_variant = None
        self._mouse_pos = (0, 0)
        self._speed_mph = 0.0
        self._is_fullscreen = False
        self._should_close_flag = False
        self._current_scene = args.scene
        self._selected_variant = args.variant
        self._has_camera_frame = False
        self._engine_active = False
        self._model_can_prewarm = False
        self._model_ready_probe: Callable[[], bool] = lambda: True
        self._scene_selection_locked_probe: Callable[[], bool] = lambda: False
        self._postprocess_preset = ""
        self._postprocess_enabled = False
        self._postprocess_callback: Callable[[bool], None] = lambda enabled: None
        self._pending_scene_change = None
        self._pending_exit_scene = False
        self._pending_drive_releases: dict[str, float] = {}

    @property
    def should_close(self) -> bool:
        return self._should_close_flag

    def process_events(self) -> None:
        if self._keyboard.consume_exit_scene_request():
            self.exit_scene()

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if view_mode == "physx" and frame.physx_debug is None:
            return
        ego_dimensions = getattr(frame.physx_debug, "ego_dimensions_lwh", None)
        if ego_dimensions is not None:
            self._latest_ego_dimensions_lwh = ego_dimensions
        rgb = select_presented_rgb(
            frame,
            view_mode,
            width=self._raster.width,
            height=self._raster.height,
        )
        self._update_camera_pil(rgb)
        if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            self._resize_for_native_model_frame()
        if frame.bev_host_uint8 is not None:
            self._update_bev_pil(frame.bev_host_uint8)
        self._render_canvas(frame.status_message, force_cpu_camera=True)
        opacity = self._visual_flare.opacity()
        canvas = self._canvas_buffer[..., :3]
        if opacity > 0.0:
            canvas = darken_rgb(canvas, opacity)
        self._frame_sink(np.ascontiguousarray(canvas).copy())

    def _present_canvas(self, **_: Any) -> None:
        self._frame_sink(np.ascontiguousarray(self._canvas_buffer[..., :3]).copy())

    def _resize_for_native_model_frame(self) -> None:
        """Grow for a native-size camera while preserving the full HUD canvas."""
        source_size = self._latest_camera_src_size
        if source_size is None:
            return
        if source_size == self._auto_sized_camera_src_size:
            return
        source_width, source_height = source_size
        current_width, current_height = self._configured_size
        target_size = (
            max(current_width, source_width + HUD_PANEL_WIDTH),
            max(current_height, source_height, MIN_WINDOW_H),
        )
        self._auto_sized_camera_src_size = source_size
        if target_size == self._canvas.size:
            return
        width, height = target_size
        self._configured_size = (width, height)
        self._canvas_buffer, self._canvas = _allocate_canvas(width, height)
        self._panel_chrome_cache_key = None
        self._panel_chrome_cache = None
        self._camera_resize_cache_key = None
        self._camera_resize_cache = None
        self._bev_panel_epoch += 1
        self._bev_panel_cache_key = None
        self._bev_panel_cache = None
        self._scene_header_rect = None
        self._variant_header_rect = None
        self._postprocess_rect = None

    def browser_key(self, key: str, down: bool) -> None:
        """Apply one browser key transition through the native input state."""
        drive_key = {
            "w": "w",
            "a": "a",
            "s": "s",
            "d": "d",
            "ArrowUp": "Up",
            "ArrowDown": "Down",
            "ArrowLeft": "Left",
            "ArrowRight": "Right",
            " ": "space",
        }.get(key)
        if drive_key is not None:
            if down:
                self._pending_drive_releases.pop(drive_key, None)
                self._keyboard_drive.set_key(drive_key, True)
                if drive_key == "space":
                    self._keyboard.set_key("space", True)
            else:
                self._pending_drive_releases[drive_key] = time.monotonic()
            return
        if not down:
            return
        if key == "1":
            self._keyboard.set_view_mode("model_rgb")
        elif key == "2":
            self._keyboard.set_view_mode("rgb")
        elif key == "3":
            self._keyboard.set_view_mode("physx")
        elif key.lower() == "r":
            self._keyboard.request_reset()
        elif key.lower() == "x":
            self.exit_scene()

    def browser_pointer(self, x: float, y: float, *, pressed: bool) -> None:
        """Apply a browser pointer event in normalized HUD coordinates."""
        width, height = self._canvas.size
        pos = (
            min(width - 1, max(0, round(float(x) * width))),
            min(height - 1, max(0, round(float(y) * height))),
        )
        self._mouse_pos = pos
        self._update_hover(pos)
        if pressed:
            self._handle_click(pos)
        self._render_canvas(None, force_cpu_camera=True)
        self._present_canvas()

    def close(self) -> None:
        self._should_close_flag = True
        bev_panel_exec = self._bev_panel_exec
        if bev_panel_exec is not None:
            bev_panel_exec.shutdown(wait=True, cancel_futures=True)
            self._bev_panel_exec = None
