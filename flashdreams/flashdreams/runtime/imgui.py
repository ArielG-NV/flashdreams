# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Dear ImGui Bundle host for server-side UI rendering."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Protocol

from flashdreams.runtime.inputs import UserInputEvent, UserInputSchema
from flashdreams.runtime.presentation import UIControlMailbox
from flashdreams.runtime.ui_input import (
    ImGuiInputCanonicalizer,
    ImGuiInputRouter,
    ImGuiInputSink,
    RawUIInputMailbox,
)


class ImGuiDrawDataRasterizer(Protocol):
    """Rasterize Dear ImGui draw data into an RGBA layer or texture."""

    def render(
        self,
        draw_data: object,
        *,
        width: int,
        height: int,
    ) -> object: ...


class SlangPyImGuiRasterizer:
    """Use SlangPy's built-in external ImGui draw-data renderer.

    SlangPy owns the Vulkan/D3D rendering backend; FlashDreams only supplies
    its target texture and command submission. The texture must support render
    target use and should be cleared to transparent before drawing.
    """

    def __init__(
        self,
        *,
        device: Any | None = None,
        texture: Any | None = None,
        slangpy_module: Any | None = None,
    ) -> None:
        if slangpy_module is None:
            try:
                slangpy_module = importlib.import_module("slangpy")
            except ImportError as exc:
                raise RuntimeError(
                    "SlangPy ImGui rendering requires the 'local-window' extra."
                ) from exc
        if (device is None) != (texture is None):
            raise ValueError(
                "device and texture must either both be set or both be None."
            )
        self._slangpy = slangpy_module
        self._imgui_backend = importlib.import_module("slangpy.ui.imgui_bundle")
        self._device = device
        self._texture = texture
        self._context = None if device is None else slangpy_module.ui.Context(device)

    def create_imgui_context(self, *, width: int, height: int) -> object:
        """Create an external ImGui context with SlangPy texture support."""
        self._ensure_resources(width=width, height=height)
        return self._imgui_backend.create_imgui_context(width, height)

    def render(
        self,
        draw_data: object,
        *,
        width: int,
        height: int,
    ) -> object:
        """Render one ImGui draw list into the configured RGBA texture."""
        self._ensure_resources(width=width, height=height)
        assert self._device is not None
        assert self._texture is not None
        assert self._context is not None
        encoder = self._device.create_command_encoder()
        self._imgui_backend.sync_draw_data_textures(
            self._device, self._context, draw_data
        )
        encoder.clear_texture_float(self._texture, clear_value=(0.0, 0.0, 0.0, 0.0))
        self._imgui_backend.render_imgui_draw_data(
            self._context, draw_data, self._texture, encoder
        )
        self._device.submit_command_buffer(encoder.finish())
        return self._texture

    def _ensure_resources(self, *, width: int, height: int) -> None:
        if self._device is not None:
            return
        device = self._slangpy.Device(
            type=self._slangpy.DeviceType.vulkan,
            enable_debug_layers=False,
            enable_cuda_launch_from_gfx=False,
            enable_ray_tracing=False,
        )
        texture = device.create_texture(
            format=self._slangpy.Format.rgba8_unorm,
            width=width,
            height=height,
            usage=(
                self._slangpy.TextureUsage.render_target
                | self._slangpy.TextureUsage.shader_resource
                | self._slangpy.TextureUsage.copy_source
            ),
            label="flashdreams_imgui_layer",
        )
        self._device = device
        self._texture = texture
        self._context = self._slangpy.ui.Context(device)


class ImGuiBundleInputSink(ImGuiInputSink):
    """Adapt canonical events to ``imgui_bundle.imgui.IO``."""

    def __init__(self, io: Any) -> None:
        self._io = io

    def add_mouse_pos_event(self, x: float, y: float) -> None:
        self._io.add_mouse_pos_event(x, y)

    def add_mouse_button_event(self, button: int, pressed: bool) -> None:
        self._io.add_mouse_button_event(button, pressed)

    def add_mouse_wheel_event(self, x: float, y: float) -> None:
        self._io.add_mouse_wheel_event(x, y)

    def add_key_event(self, key: object, pressed: bool) -> None:
        self._io.add_key_event(key, pressed)

    def add_input_characters_utf8(self, text: str) -> None:
        self._io.add_input_characters_utf8(text)

    def add_focus_event(self, focused: bool) -> None:
        self._io.add_focus_event(focused)

    def set_display_size(self, width: float, height: float) -> None:
        self._io.display_size = (width, height)


class ImGuiBundleUIRenderer:
    """Own one Dear ImGui context entirely on the presentation thread.

    Applications receive the complete ``imgui_bundle.imgui`` module in
    ``build_ui`` and can use any widget from the binding. Widget changes are
    published through :attr:`controls`; ``step`` reads an atomic snapshot from
    the same mailbox and never touches ImGui state.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        source_schema: UserInputSchema,
        build_ui: Callable[[Any, UIControlMailbox], None],
        rasterizer: ImGuiDrawDataRasterizer,
        controls: UIControlMailbox | None = None,
        imgui_module: Any | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("ImGui display dimensions must be > 0.")
        self.width = int(width)
        self.height = int(height)
        self.controls = controls or UIControlMailbox()
        self.raw_inputs = RawUIInputMailbox(source_schema=source_schema)
        self._build_ui = build_ui
        self._rasterizer = rasterizer
        self._imgui = imgui_module
        self._context: Any | None = None
        self._canonicalizer = ImGuiInputCanonicalizer()
        self._router: ImGuiInputRouter | None = None
        self._last_frame_time_s: float | None = None
        self._retry_first_frame = False

    def publish_raw_input(self, event: UserInputEvent) -> None:
        """Queue one target event without waiting for model generation."""
        self.raw_inputs.publish(event)

    def render_ui(
        self,
        *,
        presentation_index: int,
        presentation_time_s: float,
    ) -> object:
        """Pump raw input and rasterize one fresh Dear ImGui frame."""
        del presentation_index
        imgui = self._ensure_context()
        io = imgui.get_io()
        previous_time_s = self._last_frame_time_s
        if previous_time_s is not None:
            io.delta_time = max(1e-6, presentation_time_s - previous_time_s)
        self._last_frame_time_s = presentation_time_s
        self._route_raw_inputs()
        imgui.new_frame()
        self._build_ui(imgui, self.controls)
        imgui.render()
        rendered = self._rasterizer.render(
            imgui.get_draw_data(),
            width=self.width,
            height=self.height,
        )
        if not self._retry_first_frame:
            return rendered

        self._retry_first_frame = False
        imgui.new_frame()
        self._build_ui(imgui, self.controls)
        imgui.render()
        return self._rasterizer.render(
            imgui.get_draw_data(),
            width=self.width,
            height=self.height,
        )

    def _route_raw_inputs(self) -> None:
        raw_inputs = self.raw_inputs.drain()
        events = self._canonicalizer.canonicalize(
            raw_inputs,
            source_schema=self.raw_inputs.source_schema,
            display_size=(self.width, self.height),
        )
        assert self._router is not None
        self._router.route(events)

    def close(self) -> None:
        """Destroy the presentation-thread Dear ImGui context."""
        if self._context is not None and self._imgui is not None:
            self._imgui.destroy_context(self._context)
        self._context = None
        self._router = None

    def _ensure_context(self) -> Any:
        if self._imgui is None:
            try:
                imgui = importlib.import_module("imgui_bundle").imgui
            except ImportError as exc:
                raise RuntimeError(
                    "Server-side UI requires the FlashDreams 'ui' extra."
                ) from exc
            self._imgui = imgui
        imgui = self._imgui
        if self._context is None:
            context_factory = getattr(self._rasterizer, "create_imgui_context", None)
            self._context = (
                context_factory(width=self.width, height=self.height)
                if callable(context_factory)
                else imgui.create_context()
            )
            io = imgui.get_io()
            io.display_size = imgui.ImVec2(float(self.width), float(self.height))
            self._router = ImGuiInputRouter(
                ImGuiBundleInputSink(io),
                key_resolver=lambda key: _resolve_key(imgui, key),
            )
            if callable(context_factory):
                self._retry_first_frame = True
        return imgui


def create_slangpy_imgui_renderer(
    *,
    width: int,
    height: int,
    source_schema: UserInputSchema,
    build_ui: Callable[[Any, UIControlMailbox], None],
    controls: UIControlMailbox,
    slangpy_module: Any | None = None,
) -> ImGuiBundleUIRenderer:
    """Create a headless SlangPy-backed server-side ImGui renderer."""
    if slangpy_module is None:
        try:
            slangpy_module = importlib.import_module("slangpy")
        except ImportError as exc:
            raise RuntimeError(
                "SlangPy ImGui rendering requires the 'local-window' extra."
            ) from exc
    return ImGuiBundleUIRenderer(
        width=width,
        height=height,
        source_schema=source_schema,
        build_ui=build_ui,
        controls=controls,
        rasterizer=SlangPyImGuiRasterizer(
            slangpy_module=slangpy_module,
        ),
    )


def _resolve_key(imgui: Any, key: str) -> object:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "arrowdown": "down_arrow",
        "arrowleft": "left_arrow",
        "arrowright": "right_arrow",
        "arrowup": "up_arrow",
        "control": "left_ctrl",
        "ctrl": "left_ctrl",
        "return": "enter",
        " ": "space",
    }
    normalized = aliases.get(normalized, normalized)
    key_type = getattr(imgui, "Key")
    resolved = getattr(key_type, normalized, None)
    if resolved is None:
        raise ValueError(f"Dear ImGui does not expose key {key!r}.")
    return resolved


__all__ = [
    "ImGuiBundleInputSink",
    "ImGuiBundleUIRenderer",
    "ImGuiDrawDataRasterizer",
    "SlangPyImGuiRasterizer",
    "create_slangpy_imgui_renderer",
]
