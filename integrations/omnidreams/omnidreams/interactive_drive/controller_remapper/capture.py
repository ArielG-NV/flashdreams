# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure SDL3 semantic and raw-joystick capture used by the remapper UI."""

from __future__ import annotations

from dataclasses import dataclass

from omnidreams.interactive_drive.input.controller_profiles import (
    Binding,
    ControllerState,
    parse_joystick_control_key,
)

_AXIS_ACTIONS = {"steering", "throttle", "brake"}
_BUTTON_ACTIONS = {"reverse", "reset", "exit"}


def moved_axis(
    baseline: ControllerState,
    current: ControllerState,
    *,
    threshold: float = 0.35,
) -> str | None:
    """Return the SDL3 axis with the largest intentional movement."""
    changes = sorted(
        (
            (abs(value - baseline.axes.get(axis, 0.0)), axis)
            for axis, value in current.axes.items()
        ),
        reverse=True,
    )
    if not changes or changes[0][0] < threshold:
        return None
    return changes[0][1]


def captured_binding(
    action: str,
    baseline: ControllerState,
    current: ControllerState,
    *,
    last_button_down: str | None = None,
    threshold: float = 0.35,
) -> Binding | None:
    """Resolve a remapping gesture to a semantic or raw SDL3 binding."""
    if action in _BUTTON_ACTIONS:
        if last_button_down is None:
            return None
        raw = parse_joystick_control_key(last_button_down)
        return (
            Binding("button", last_button_down)
            if raw is None
            else Binding("button", raw[1], device=raw[0])
        )
    if action not in _AXIS_ACTIONS:
        raise ValueError(f"Unknown driving action: {action!r}")

    # Throttle and brake may be digital (Switch Pro ZL/ZR, for example).
    if action != "steering" and last_button_down is not None:
        raw = parse_joystick_control_key(last_button_down)
        return (
            Binding("button", last_button_down)
            if raw is None
            else Binding("button", raw[1], device=raw[0])
        )
    axis = moved_axis(baseline, current, threshold=threshold)
    if axis is None:
        return None
    raw = parse_joystick_control_key(axis)
    if raw is None:
        return Binding("axis", axis)
    rest = baseline.axes.get(axis, 0.0) if action in {"throttle", "brake"} else None
    return Binding(
        "axis",
        raw[1],
        device=raw[0],
        rest=rest,
        invert=action in {"throttle", "brake"} and current.axes[axis] < rest,
    )


@dataclass
class CaptureSession:
    """One in-progress semantic remap gesture; no files or threads involved."""

    action: str | None = None
    baseline: ControllerState | None = None

    @property
    def listening(self) -> bool:
        return self.action is not None and self.baseline is not None

    def start(self, action: str, state: ControllerState) -> None:
        self.action = action
        self.baseline = state.copy()

    def cancel(self) -> None:
        self.action = None
        self.baseline = None

    def feed(
        self,
        state: ControllerState,
        *,
        last_button_down: str | None = None,
    ) -> tuple[str, Binding] | None:
        if not self.listening:
            return None
        assert self.action is not None and self.baseline is not None
        binding = captured_binding(
            self.action,
            self.baseline,
            state,
            last_button_down=last_button_down,
        )
        if binding is None:
            return None
        action = self.action
        self.cancel()
        return action, binding
