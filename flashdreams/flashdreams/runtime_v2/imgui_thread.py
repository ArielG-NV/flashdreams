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

"""Dear ImGui worker and SlangPy Vulkan/CUDA renderer."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, cast, final

import torch
from torch import Tensor

from flashdreams.api_v2.thread import UIThread
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

StateT = TypeVar("StateT")


class ImGUIThread(UIThread[StateT], ABC):
    """UI worker rendered through SlangPy with Vulkan/CUDA interop."""

    def __init__(
        self,
        *,
        state: StateT,
        frequency: int,
        output_layout: VideoTensorLayout,
        width: int,
        height: int,
    ) -> None:
        """Initialize a SlangPy ImGui worker without creating GPU resources.

        Args:
            state: Mutable state owned by this worker.
            frequency: Maximum UI frames per second.
            output_layout: Layout used to wrap the rendered frame.
            width: Render target width in pixels.
            height: Render target height in pixels.
        """
        super().__init__(
            state=state,
            frequency=frequency,
            output_layout=output_layout,
        )
        self._renderer = SlangPyImGUIRenderer(width=width, height=height)

    @abstractmethod
    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> None:
        """Issue one frame of Dear ImGui widget calls.

        Args:
            imgui: Dear ImGui module for issuing widget calls.
            step_index: Zero-based index of the UI step being rendered.
            events: Input events collected for this UI step.
        """
        ...

    @final
    def step_ui(self, step_index: int, events: UserInputEvents) -> Tensor:
        """Route input and queue one UI frame for rendering."""

        def draw_ui(imgui: Any) -> None:
            self.draw_ui(imgui, step_index, events)

        return self._renderer.render(events, draw_ui)

    @final
    def wait_for_ui_to_render(self, frame: Tensor) -> Tensor:
        """Return the CUDA frame after the renderer signals completion."""
        return self._renderer.wait_for_cuda_frame(frame)

    @final
    def draw_frame(
        self,
        imgui: Any,
        frame: Tensor,
        size: tuple[float, float],
    ) -> None:
        """Draw a normalized video frame as a Dear ImGui image."""
        self._renderer.draw_frame(imgui, frame, size)

    def reset(self) -> None:
        """Reset renderer input state for a new session generation."""
        self._renderer.reset()

    def _close(self) -> None:
        """Release renderer resources on the UI worker thread."""
        self._renderer.close()


class SlangPyImGUIRenderer:
    """Render external Dear ImGui draw data through SlangPy and CUDA interop."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        slangpy_module: Any | None = None,
        imgui_module: Any | None = None,
    ) -> None:
        """Configure a renderer whose native resources are created lazily.

        Args:
            width: Render target width in pixels.
            height: Render target height in pixels.
            slangpy_module: Optional injected SlangPy module for tests.
            imgui_module: Optional injected ``imgui_bundle.imgui`` module for tests.

        Raises:
            ValueError: A render dimension is not positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError("ImGui render dimensions must be > 0.")
        self.width = int(width)
        self.height = int(height)
        self._slangpy = slangpy_module
        self._imgui = imgui_module
        self._imgui_backend: Any | None = None
        self._device: Any | None = None
        self._ui_context: Any | None = None
        self._imgui_context: Any | None = None
        self._target: Any | None = None
        self._rgba_buffer: Any | None = None
        self._rgba_tensor: Tensor | None = None
        self._rgba_buffer_size = 0
        self._rgba_row_pitch = 0
        self._image_texture: Any | None = None
        self._image_texture_ref: Any | None = None
        self._has_rendered = False

    def render(
        self,
        events: UserInputEvents,
        draw_ui: Callable[[Any], None],
    ) -> Tensor:
        """Queue input and render one ImGui frame into shared RGBA storage."""
        self._ensure_initialized()
        assert self._device is not None
        assert self._imgui is not None
        assert self._imgui_backend is not None
        assert self._ui_context is not None
        assert self._imgui_context is not None
        assert self._target is not None
        assert self._rgba_buffer is not None
        assert self._rgba_tensor is not None

        self._imgui.set_current_context(self._imgui_context)
        if self._has_rendered:
            self._device.sync_to_cuda(_current_cuda_stream())
        _route_input_events(
            events,
            io=self._imgui.get_io(),
            imgui=self._imgui,
            width=self.width,
            height=self.height,
        )
        self._imgui.new_frame()
        draw_ui(self._imgui)
        self._imgui.render()
        draw_data = self._imgui.get_draw_data()
        self._imgui_backend.sync_draw_data_textures(
            self._device,
            self._ui_context,
            draw_data,
        )

        encoder = self._device.create_command_encoder()
        encoder.clear_texture_float(
            self._target,
            clear_value=(0.0, 0.0, 0.0, 0.0),
        )
        self._imgui_backend.render_imgui_draw_data(
            self._ui_context,
            draw_data,
            self._target,
            encoder,
        )
        encoder.copy_texture_to_buffer(
            self._rgba_buffer,
            0,
            self._rgba_buffer_size,
            self._rgba_row_pitch,
            self._target,
            0,
            0,
            [0, 0, 0],
            [self.width, self.height, 1],
        )
        self._device.submit_command_buffer(encoder.finish())
        self._has_rendered = True
        return self._rgba_tensor

    def wait_for_cuda_frame(self, frame: Tensor) -> Tensor:
        """Wait for Vulkan and convert shared RGBA bytes for compositing."""
        if self._device is None:
            raise RuntimeError("Render a frame before waiting for it.")
        self._device.sync_to_device(_current_cuda_stream())
        rgba = frame.permute(2, 0, 1).to(torch.float32)
        color = rgba[:3].mul_(2.0 / 255.0).sub_(1.0)
        alpha = rgba[3:4].mul_(1.0 / 255.0)
        return torch.cat((color, alpha), dim=0).unsqueeze(0)

    def draw_frame(
        self,
        imgui: Any,
        frame: Tensor,
        size: tuple[float, float],
    ) -> None:
        """Draw a video frame as a Dear ImGui image.

        Args:
            imgui: Dear ImGui module used by the active render callback.
            frame: Read-only ``[C, H, W]`` frame with color in ``[-1, 1]``.
            size: Display width and height in pixels.

        Raises:
            RuntimeError: Called outside an active renderer callback.
            ValueError: ``frame`` is not a supported image shape.
        """
        if self._device is None or self._imgui_backend is None:
            raise RuntimeError("draw_frame() must be called from draw_ui().")
        rgba = _frame_to_rgba8(frame)
        height, width = rgba.shape[:2]
        texture = self._image_texture
        if texture is None or (texture.width, texture.height) != (width, height):
            self._release_image_texture()
            assert self._slangpy is not None
            texture = self._device.create_texture(
                format=self._slangpy.Format.rgba8_unorm,
                width=width,
                height=height,
                usage=self._slangpy.TextureUsage.shader_resource,
                data=rgba.numpy(),
                label="flashdreams_imgui_image",
            )
            self._image_texture = texture
            self._image_texture_ref = self._imgui_backend.texture_ref(texture)
        else:
            texture.copy_from_numpy(rgba.numpy())
        imgui.image(self._image_texture_ref, size)

    def reset(self) -> None:
        """Release held input state before the next generation."""
        if self._imgui is None or self._imgui_context is None:
            return
        self._imgui.set_current_context(self._imgui_context)
        io = self._imgui.get_io()
        io.add_focus_event(False)
        io.add_focus_event(True)

    def close(self) -> None:
        """Destroy the ImGui context after pending GPU work completes."""
        if self._device is not None:
            torch.cuda.current_stream().synchronize()
            self._device.wait_for_idle()
        if self._imgui is not None and self._imgui_context is not None:
            self._imgui.destroy_context(self._imgui_context)
        self._release_image_texture()
        self._imgui_context = None
        self._rgba_tensor = None
        self._rgba_buffer = None
        self._rgba_buffer_size = 0
        self._rgba_row_pitch = 0
        self._target = None
        self._ui_context = None
        self._device = None

    def _release_image_texture(self) -> None:
        if self._image_texture_ref is not None and self._imgui_backend is not None:
            self._imgui_backend._release_texture(self._image_texture_ref.get_tex_id())
        self._image_texture_ref = None
        self._image_texture = None

    def _ensure_initialized(self) -> None:
        if self._device is not None:
            return
        slangpy = self._slangpy
        if slangpy is None:
            try:
                slangpy = importlib.import_module("slangpy")
            except ImportError as error:
                raise RuntimeError(
                    "ImGui rendering requires the FlashDreams 'local-window' extra."
                ) from error
            self._slangpy = slangpy
        imgui = self._imgui
        if imgui is None:
            try:
                imgui = importlib.import_module("imgui_bundle").imgui
            except ImportError as error:
                raise RuntimeError(
                    "ImGui rendering requires the FlashDreams 'ui' extra."
                ) from error
            self._imgui = imgui
        self._imgui_backend = importlib.import_module("slangpy.ui.imgui_bundle")

        if not torch.cuda.is_available():
            raise RuntimeError("SlangPy ImGui rendering requires CUDA.")
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        torch.cuda.set_device(torch.cuda.current_device())
        torch.cuda.current_stream()
        existing_handles = list(slangpy.get_cuda_current_context_native_handles())
        if not existing_handles:
            raise RuntimeError("Could not obtain the current CUDA context handles.")
        device = slangpy.Device(
            type=slangpy.DeviceType.vulkan,
            enable_debug_layers=False,
            enable_cuda_interop=True,
            enable_cuda_launch_from_gfx=False,
            enable_ray_tracing=False,
            existing_device_handles=existing_handles,
        )
        if not device.supports_cuda_interop:
            raise RuntimeError("The Vulkan device does not support CUDA interop.")

        target = device.create_texture(
            format=slangpy.Format.rgba8_unorm,
            width=self.width,
            height=self.height,
            usage=(
                slangpy.TextureUsage.render_target
                | slangpy.TextureUsage.shader_resource
                | slangpy.TextureUsage.copy_source
            ),
            label="flashdreams_imgui_target",
        )
        layout = target.get_subresource_layout(0)
        size_bytes = int(layout.size_in_bytes)
        row_pitch = int(layout.row_pitch)
        rgba_buffer = device.create_buffer(
            size=size_bytes,
            usage=slangpy.BufferUsage.shared | slangpy.BufferUsage.copy_destination,
            label="flashdreams_imgui_rgba",
        )
        rgba_tensor = cast(
            Tensor,
            rgba_buffer.to_torch(
                type=slangpy.DataType.uint8,
                shape=[self.height, self.width, 4],
                strides=[row_pitch, 4, 1],
            ),
        )

        self._device = device
        self._ui_context = slangpy.ui.Context(device)
        self._imgui_context = self._imgui_backend.create_imgui_context(
            self.width,
            self.height,
        )
        assert self._imgui is not None
        self._imgui.get_io().set_ini_filename("")
        self._target = target
        self._rgba_buffer = rgba_buffer
        self._rgba_tensor = rgba_tensor
        self._rgba_buffer_size = size_bytes
        self._rgba_row_pitch = row_pitch


def _route_input_events(
    events: UserInputEvents,
    *,
    io: Any,
    imgui: Any,
    width: int,
    height: int,
) -> None:
    """Route supported runtime input events into Dear ImGui IO."""
    for event in events.get_events():
        data = event.get_event_data()
        if isinstance(data, KeyboardUserInputEventData):
            key = _resolve_imgui_key(imgui, data.key)
            if key is not None:
                io.add_key_event(key, data.pressed)
            if data.pressed and len(data.key) == 1:
                io.add_input_characters_utf8(data.key)
        elif isinstance(data, FocusUserInputEventData):
            io.add_focus_event(data.focused)
        elif isinstance(data, MouseUserInputEventData):
            io.add_mouse_pos_event(data.x * width, data.y * height)
            if data.action == "button":
                io.add_mouse_button_event(data.button, data.pressed)
            elif data.action == "wheel":
                io.add_mouse_wheel_event(data.wheel_x, data.wheel_y)


def _frame_to_rgba8(frame: Tensor) -> Tensor:
    """Convert one normalized video frame to CPU ``[H, W, 4]`` bytes."""
    if frame.ndim != 3 or frame.shape[0] not in (1, 3, 4):
        raise ValueError("An ImGui image must have shape [C, H, W] for C in {1, 3, 4}.")
    color = frame[:3]
    if color.shape[0] == 1:
        color = color.repeat(3, 1, 1)
    color = color.to(torch.float32).clamp(-1.0, 1.0).add(1.0).mul(127.5)
    if frame.shape[0] == 4:
        alpha = frame[3:4].to(torch.float32).clamp(0.0, 1.0).mul(255.0)
    else:
        alpha = torch.full_like(color[:1], 255.0)
    return (
        torch.cat((color, alpha), dim=0)
        .permute(1, 2, 0)
        .round_()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )


def _resolve_imgui_key(imgui: Any, key: str) -> Any | None:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        " ": "space",
        "alt": "left_alt",
        "arrowdown": "down_arrow",
        "arrowleft": "left_arrow",
        "arrowright": "right_arrow",
        "arrowup": "up_arrow",
        "control": "left_ctrl",
        "ctrl": "left_ctrl",
        "esc": "escape",
        "meta": "left_super",
        "return": "enter",
        "shift": "left_shift",
    }
    punctuation = {
        "'": "apostrophe",
        ",": "comma",
        ".": "period",
        "/": "slash",
        ";": "semicolon",
        "=": "equal",
        "[": "left_bracket",
        "\\": "backslash",
        "]": "right_bracket",
        "`": "grave_accent",
    }
    normalized = aliases.get(key.lower(), aliases.get(normalized, normalized))
    normalized = punctuation.get(key, normalized)
    if len(normalized) == 1 and normalized.isdigit():
        normalized = f"_{normalized}"
    return getattr(imgui.Key, normalized, None)


def _current_cuda_stream() -> int:
    """Return the current PyTorch CUDA stream handle."""
    return int(torch.cuda.current_stream().cuda_stream)


__all__ = ["ImGUIThread", "SlangPyImGUIRenderer"]
