# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation-only UI for the reusable interactive driving application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from flashdreams.demo import (
    HWC_UINT8_FORMAT,
    PostProcessingFrame,
    PostProcessingInput,
    PostProcessingOutput,
    PostProcessingPipeline,
    PostProcessingPipelineStep,
    frame_tensor_to_hwc_uint8,
)

_NVIDIA_GREEN = (118, 185, 0)
_PANEL = (18, 20, 27)
_PANEL_EDGE = (65, 70, 82)
_TEXT = (238, 240, 244)
_MUTED = (145, 151, 164)
_ACTIVE = (118, 185, 0)
_INACTIVE = (52, 57, 68)


@dataclass(frozen=True, kw_only=True, slots=True)
class InteractiveDriveUIState:
    """Immutable control snapshot rendered over one generated chunk."""

    throttle: float = 0.0
    """Normalized accelerator level in ``[0, 1]``."""

    brake: float = 0.0
    """Normalized brake level in ``[0, 1]``."""

    steer: float = 0.0
    """Normalized steering level in ``[-1, 1]``."""

    reverse: bool = False
    """Whether the current throttle command drives in reverse."""

    source: str = "keyboard"
    """Short input-source label rendered in the HUD."""

    title: str = "INTERACTIVE DRIVE"
    """Integration-provided title rendered in the HUD."""

    @classmethod
    def from_driver_command(
        cls,
        command: Mapping[str, Any],
        *,
        source: str = "keyboard",
        title: str = "INTERACTIVE DRIVE",
    ) -> "InteractiveDriveUIState":
        """Create a clamped UI snapshot from a canonical driver command."""
        return cls(
            throttle=_clamp(float(command.get("throttle", 0.0)), 0.0, 1.0),
            brake=_clamp(float(command.get("brake", 0.0)), 0.0, 1.0),
            steer=_clamp(float(command.get("steer", 0.0)), -1.0, 1.0),
            reverse=bool(command.get("reverse", False)),
            source=source.strip() or "input",
            title=title.strip() or "INTERACTIVE DRIVE",
        )


def build_interactive_drive_ui_pipeline(
    state: InteractiveDriveUIState,
) -> PostProcessingPipeline:
    """Build the frame-scoped UI pipeline for a generated driving chunk."""
    return PostProcessingPipeline(
        (
            PostProcessingPipelineStep(
                input_kind="frame",
                operation=lambda partition: _render_ui(partition, state=state),
                name="interactive-drive-ui",
            ),
        )
    )


def _render_ui(
    partition: PostProcessingInput,
    *,
    state: InteractiveDriveUIState,
) -> PostProcessingOutput:
    if not isinstance(partition, PostProcessingFrame):
        raise TypeError("The interactive-drive UI requires a frame partition.")
    if partition.format.layout not in ("chw", "hwc"):
        raise ValueError(
            "The interactive-drive UI requires CHW or HWC frames, "
            f"got {partition.format.layout!r}."
        )

    frame = frame_tensor_to_hwc_uint8(
        partition.data,
        layout=partition.format.layout,
        value_range=partition.format.value_range,
    )
    host_frame = np.array(frame.detach().cpu().numpy(), dtype=np.uint8, copy=True)
    image = Image.fromarray(host_frame, mode="RGB")
    _draw_ui(image, state=state)
    output = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    return PostProcessingOutput(data=output, format=HWC_UINT8_FORMAT)


def _draw_ui(image: Image.Image, *, state: InteractiveDriveUIState) -> None:
    draw = ImageDraw.Draw(image, "RGB")
    width, height = image.size
    scale = max(0.65, min(width / 1280.0, height / 704.0))
    margin = max(8, round(18 * scale))
    panel_width = min(width - margin * 2, max(250, round(360 * scale)))
    panel_height = min(height - margin * 2, max(116, round(150 * scale)))
    left = margin
    top = height - margin - panel_height
    right = left + panel_width
    bottom = top + panel_height
    radius = max(8, round(14 * scale))
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=radius,
        fill=_PANEL,
        outline=_PANEL_EDGE,
        width=max(1, round(2 * scale)),
    )

    font = ImageFont.load_default()
    draw.rectangle(
        (
            left + margin,
            top + margin,
            left + margin + max(4, round(5 * scale)),
            top + margin + round(18 * scale),
        ),
        fill=_NVIDIA_GREEN,
    )
    draw.text(
        (left + margin + round(14 * scale), top + margin),
        state.title.upper()[:36],
        fill=_TEXT,
        font=font,
    )
    draw.text(
        (right - margin - max(40, round(64 * scale)), top + margin),
        state.source.upper()[:12],
        fill=_MUTED,
        font=font,
    )

    control_top = top + round(48 * scale)
    box = max(25, round(31 * scale))
    gap = max(5, round(7 * scale))
    controls = (
        ("W", state.throttle > 0.01),
        ("A", state.steer > 0.01),
        ("S", state.brake > 0.01 or state.reverse),
        ("D", state.steer < -0.01),
    )
    for index, (label, active) in enumerate(controls):
        x0 = left + margin + index * (box + gap)
        draw.rounded_rectangle(
            (x0, control_top, x0 + box, control_top + box),
            radius=max(3, round(5 * scale)),
            fill=_ACTIVE if active else _INACTIVE,
        )
        _draw_centered_text(
            draw, (x0, control_top, x0 + box, control_top + box), label, font
        )

    bar_left = left + margin
    bar_right = right - margin
    bar_top = bottom - margin - max(14, round(18 * scale))
    bar_bottom = bottom - margin
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_right, bar_bottom),
        radius=max(3, round(6 * scale)),
        fill=_INACTIVE,
    )
    center = (bar_left + bar_right) // 2
    marker_x = round(center - state.steer * (bar_right - bar_left) * 0.45)
    marker_half = max(2, round(4 * scale))
    draw.rounded_rectangle(
        (marker_x - marker_half, bar_top, marker_x + marker_half, bar_bottom),
        radius=marker_half,
        fill=_NVIDIA_GREEN,
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    font: Any,
) -> None:
    left, top, right, bottom = bounds
    text_bounds = draw.textbbox((0, 0), text, font=font)
    text_width = text_bounds[2] - text_bounds[0]
    text_height = text_bounds[3] - text_bounds[1]
    draw.text(
        (
            left + (right - left - text_width) / 2 - text_bounds[0],
            top + (bottom - top - text_height) / 2 - text_bounds[1],
        ),
        text,
        fill=_TEXT,
        font=font,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = [
    "InteractiveDriveUIState",
    "build_interactive_drive_ui_pipeline",
]
