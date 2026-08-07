# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure SDL3 remapping gesture capture."""

from __future__ import annotations

import pytest
from omnidreams.interactive_drive.input.wheel_profiles import (
    Binding,
    ControllerState,
)
from omnidreams.interactive_drive.input_config.capture import (
    CaptureSession,
    captured_binding,
    moved_axis,
)
from omnidreams.interactive_drive.input_config.app import _ui_scale_for_display

pytestmark = pytest.mark.ci_cpu


def _state(**axes) -> ControllerState:
    values = {
        "left_x": 0.0,
        "left_y": 0.0,
        "right_x": 0.0,
        "right_y": 0.0,
        "left_trigger": 0.0,
        "right_trigger": 0.0,
    }
    values.update(axes)
    return ControllerState(connected=True, axes=values)


def test_moved_axis_selects_largest_semantic_change() -> None:
    assert moved_axis(_state(), _state(left_x=0.5, right_x=0.8)) == "right_x"


def test_moved_axis_ignores_stick_noise() -> None:
    assert moved_axis(_state(), _state(left_x=0.05)) is None


def test_steering_capture_requires_an_axis() -> None:
    assert captured_binding(
        "steering", _state(), _state(left_x=0.8), last_button_down="a"
    ) == Binding("axis", "left_x")


def test_switch_pro_digital_trigger_can_bind_throttle() -> None:
    assert captured_binding(
        "throttle", _state(), _state(), last_button_down="right_bumper"
    ) == Binding("button", "right_bumper")


def test_action_capture_uses_button_edge() -> None:
    assert captured_binding(
        "reset", _state(), _state(), last_button_down="start"
    ) == Binding("button", "start")


def test_capture_session_finishes_after_one_gesture() -> None:
    session = CaptureSession()
    session.start("brake", _state())
    assert session.feed(_state(left_trigger=0.2)) is None
    assert session.feed(_state(left_trigger=1.0)) == (
        "brake",
        Binding("axis", "left_trigger"),
    )
    assert session.listening is False


def test_separate_pedal_axis_capture_keeps_device_and_released_position() -> None:
    baseline = ControllerState(
        connected=True,
        input_kind="joystick",
        axes={"d0:axis_0": 0.0, "d1:axis_2": 1.0},
    )
    pressed = ControllerState(
        connected=True,
        input_kind="joystick",
        axes={"d0:axis_0": 0.01, "d1:axis_2": -0.75},
    )
    assert captured_binding("throttle", baseline, pressed) == Binding(
        "axis", "axis_2", device=1, rest=1.0, invert=True
    )


def test_raw_wheel_button_capture_keeps_device_index() -> None:
    state = ControllerState(connected=True, input_kind="joystick")
    assert captured_binding(
        "reset", state, state, last_button_down="d2:button_7"
    ) == Binding("button", "button_7", device=2)


def test_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown driving action"):
        captured_binding("horn", _state(), _state())


@pytest.mark.parametrize(
    ("width", "height", "dpi", "expected"),
    (
        (1920, 1080, 96.0, 1.0),
        (3840, 2160, 144.0, 1.5),
        (3840, 2160, 192.0, 2.0),
        (1366, 768, 144.0, 1.0),
        (7680, 4320, 384.0, 3.0),
    ),
)
def test_ui_scale_uses_resolution_dpi_and_screen_fit(
    width: int, height: int, dpi: float, expected: float
) -> None:
    assert _ui_scale_for_display(width, height, dpi) == pytest.approx(expected)
