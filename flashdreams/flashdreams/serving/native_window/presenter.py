# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy native video presenter."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class SlangPyNativePresenter:
    """Present RGB frames and forward driving keys."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        title: str,
        on_key: Callable[[str, str], None],
    ) -> None:
        try:
            self.spy = importlib.import_module("slangpy")
        except ImportError as exc:
            raise RuntimeError("Native-window output requires SlangPy.") from exc
        self.width = width
        self.height = height
        self.on_key = on_key
        self.closed = False
        self.window = self.spy.Window(
            width=width, height=height, title=title, resizable=False
        )
        self.device = self.spy.Device(
            type=self.spy.DeviceType.vulkan,
            enable_debug_layers=False,
            enable_cuda_launch_from_gfx=False,
            enable_ray_tracing=False,
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(
            width=width, height=height, format=self._surface_format()
        )
        self.texture = self.device.create_texture(
            format=self.spy.Format.rgba8_unorm,
            width=width,
            height=height,
            usage=(
                self.spy.TextureUsage.shader_resource
                | self.spy.TextureUsage.unordered_access
                | self.spy.TextureUsage.copy_destination
            ),
        )
        self.keys = self._key_map()
        self.escape = getattr(self.spy.KeyCode, "escape", None)
        self.window.on_keyboard_event = self._on_keyboard

    @property
    def should_close(self) -> bool:
        return self.closed or self.window.should_close()

    def process_events(self) -> None:
        self.window.process_events()

    def present_frame(self, frame: object) -> None:
        target = self.surface.acquire_next_image()
        if not target:
            time.sleep(0.001)
            return
        rgb = _as_rgb(frame)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Native frame shape {rgb.shape} does not match "
                f"{(self.height, self.width, 3)}."
            )
        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        rgba[..., :3], rgba[..., 3] = rgb, 255
        self.texture.copy_from_numpy(rgba)
        encoder = self.device.create_command_encoder()
        encoder.blit(target, self.texture)
        self.device.submit_command_buffer(encoder.finish())
        self.surface.present()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.window.close()

    def _surface_format(self) -> Any:
        supported = self.surface.info.formats
        for name in ("rgba8_unorm", "bgra8_unorm", "bgrx8_unorm"):
            value = getattr(self.spy.Format, name)
            if value in supported:
                return value
        raise RuntimeError("Native output requires a linear surface format.")

    def _key_map(self) -> dict[object, str]:
        keys: dict[object, str] = {}
        for action, names in {
            "w": ("w", "up", "arrow_up"),
            "a": ("a", "left", "arrow_left"),
            "s": ("s", "down", "arrow_down"),
            "d": ("d", "right", "arrow_right"),
        }.items():
            for name in names:
                value = getattr(self.spy.KeyCode, name, None)
                if value is not None:
                    keys[value] = action
        return keys

    def _on_keyboard(self, event: Any) -> None:
        pressed = event.is_key_press()
        released = event.is_key_release()
        if pressed and event.key == self.escape:
            self.close()
            return
        key = self.keys.get(event.key)
        if key is not None and (pressed or released):
            self.on_key("keydown" if pressed else "keyup", key)


def _as_rgb(frame: object) -> np.ndarray:
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])


__all__ = ["SlangPyNativePresenter"]
