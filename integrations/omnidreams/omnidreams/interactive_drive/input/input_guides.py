# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Human-readable keyboard and SDL3 controller/wheel mappings."""

from __future__ import annotations

from typing import Any

from omnidreams.interactive_drive.input.controller_profiles import (
    Binding,
    ControllerProfile,
)

KEYBOARD_GUIDE: tuple[tuple[str, str], ...] = (
    ("W / Up", "Throttle"),
    ("S / Down / Space", "Brake"),
    ("A / D / Left / Right", "Steer"),
    ("R", "Reset / respawn"),
    ("X", "Exit scene"),
    ("1 / 2 / 3", "Camera view"),
)

_CONTROL_LABELS = {
    "left_x": "Left stick X",
    "left_y": "Left stick Y",
    "right_x": "Right stick X",
    "right_y": "Right stick Y",
    "left_trigger": "Left trigger",
    "right_trigger": "Right trigger",
    "a": "A / South",
    "b": "B / East",
    "x": "X / West",
    "y": "Y / North",
    "left_bumper": "Left bumper",
    "right_bumper": "Right bumper",
    "back": "Back / −",
    "start": "Start / +",
    "guide": "Guide / Home",
    "left_thumb": "Left stick press",
    "right_thumb": "Right stick press",
    "up": "D-pad up",
    "right": "D-pad right",
    "down": "D-pad down",
    "left": "D-pad left",
}

_NINTENDO_FACE_BUTTON_LABELS = {
    "a": "A / East",
    "b": "B / South",
    "x": "X / North",
    "y": "Y / West",
}


def _binding_label(profile: ControllerProfile, binding: Binding | None) -> str | None:
    if binding is None:
        return None
    labels = (
        _NINTENDO_FACE_BUTTON_LABELS
        if profile.swap_face_buttons
        else _CONTROL_LABELS
    )
    control = labels.get(
        binding.control,
        _CONTROL_LABELS.get(
            binding.control, binding.control.replace("_", " ").title()
        ),
    )
    if binding.is_raw_joystick and binding.device < len(profile.devices):
        return f"{profile.devices[binding.device].name}: {control}"
    return control


def profile_guide(profile: ControllerProfile) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for action, label in (
        ("steering", "Steer"),
        ("throttle", "Throttle"),
        ("brake", "Brake"),
        ("reverse", "Toggle reverse"),
        ("reset", "Reset / respawn"),
        ("exit", "Exit scene"),
    ):
        control = _binding_label(profile, profile.binding(action))
        if control is not None:
            rows.append((control, label))
    return tuple(rows)


def active_input_guide(
    controller_input: Any | None,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    if controller_input is not None and controller_input.state.connected:
        kind = "wheel" if controller_input.profile.is_joystick else "controller"
        return (
            kind,
            controller_input.profile.display_name,
            profile_guide(controller_input.profile),
        )
    return "keyboard", "Keyboard", KEYBOARD_GUIDE
