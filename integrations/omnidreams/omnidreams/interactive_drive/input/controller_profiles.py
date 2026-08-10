# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SDL3 gamepad and generic joystick profiles for interactive-drive.

Console-style controllers use SlangPy's SDL3 gamepad callbacks. Racing wheels,
pedal sets, flight controls, and other arbitrary devices use SDL3's lower-level
Joystick API through :mod:`.sdl3_joystick`. Both backends share one portable
profile format and never expose evdev paths, WinMM slots, or HID reports.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

SDL3_AXES: tuple[str, ...] = (
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "left_trigger",
    "right_trigger",
)

SDL3_BUTTONS: tuple[str, ...] = (
    "a",
    "b",
    "x",
    "y",
    "left_bumper",
    "right_bumper",
    "back",
    "start",
    "guide",
    "left_thumb",
    "right_thumb",
    "up",
    "right",
    "down",
    "left",
)

DRIVE_ACTIONS: tuple[str, ...] = (
    "steering",
    "throttle",
    "brake",
    "reverse",
    "reset",
    "exit",
)

GAMEPAD_BACKEND = "sdl3-gamepad"
JOYSTICK_BACKEND = "sdl3-joystick"

FFB_MODES: tuple[str, ...] = ("auto", "autocenter", "constant_force")
"""Supported SDL3 wheel force-feedback strategies."""

_DEFAULT_BINDINGS = {
    "steering": ("axis", "left_x"),
    "throttle": ("axis", "right_trigger"),
    "brake": ("axis", "left_trigger"),
    "reverse": ("button", "b"),
    "reset": ("button", "y"),
    "exit": ("button", "back"),
}

_SWAPPED_FACE_BUTTONS = {"a": "b", "b": "a", "x": "y", "y": "x"}
_JOYSTICK_AXIS_RE = re.compile(r"axis_(\d+)$")
_JOYSTICK_BUTTON_RE = re.compile(r"button_(\d+)$")


@dataclass(frozen=True)
class DeviceSpec:
    """Stable description of one SDL3 joystick used by a profile.

    GUID matching is preferred for a local profile. VID/PID and exact device
    name are retained as fallbacks because SDL documents GUIDs as
    platform-dependent. ``ordinal`` disambiguates two identical devices.
    """

    name: str
    guid: str = ""
    vendor_id: int = 0
    product_id: int = 0
    kind: str = "unknown"
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SDL3 joystick device name cannot be empty")
        if self.vendor_id < 0 or self.product_id < 0 or self.ordinal < 0:
            raise ValueError("SDL3 joystick identifiers must be non-negative")


@dataclass(frozen=True)
class Binding:
    """An SDL3 semantic control or a raw joystick axis/button.

    ``device`` indexes :attr:`ControllerProfile.devices` for joystick bindings.
    ``rest`` stores a pedal's released value in SDL's normalized ``[-1, 1]``
    range; ``invert`` means engagement moves toward ``-1``.
    """

    kind: str
    control: str
    device: int = 0
    rest: float | None = None
    invert: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"axis", "button"}:
            raise ValueError(f"Unknown SDL3 binding type: {self.kind!r}")
        semantic = SDL3_AXES if self.kind == "axis" else SDL3_BUTTONS
        raw_pattern = _JOYSTICK_AXIS_RE if self.kind == "axis" else _JOYSTICK_BUTTON_RE
        if self.control not in semantic and raw_pattern.fullmatch(self.control) is None:
            raise ValueError(f"Unknown SDL3 {self.kind}: {self.control!r}")
        if self.device < 0:
            raise ValueError("SDL3 joystick device index must be non-negative")
        if self.rest is not None and not -1.0 <= float(self.rest) <= 1.0:
            raise ValueError("SDL3 joystick rest value must be in [-1, 1]")

    @property
    def raw_index(self) -> int | None:
        pattern = _JOYSTICK_AXIS_RE if self.kind == "axis" else _JOYSTICK_BUTTON_RE
        match = pattern.fullmatch(self.control)
        return None if match is None else int(match.group(1))

    @property
    def is_raw_joystick(self) -> bool:
        return self.raw_index is not None


@dataclass(frozen=True)
class ControllerProfile:
    """Driving input profile for either SDL3 Gamepad or Joystick input."""

    name: str
    display_name: str
    bindings: dict[str, Binding] = field(default_factory=dict)
    backend: str = GAMEPAD_BACKEND
    devices: tuple[DeviceSpec, ...] = ()
    swap_face_buttons: bool = False
    invert_steering: bool = False
    steering_range: float = 0.75
    steering_deadzone: float = 0.08
    ffb_enabled: bool = False
    ffb_mode: str = "auto"
    ffb_gain: float = 0.5
    is_default: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {GAMEPAD_BACKEND, JOYSTICK_BACKEND}:
            raise ValueError(f"Unknown SDL3 input backend: {self.backend!r}")
        if self.ffb_mode not in FFB_MODES:
            raise ValueError(f"Unknown SDL3 force-feedback mode: {self.ffb_mode!r}")
        unknown_actions = set(self.bindings) - set(DRIVE_ACTIONS)
        if unknown_actions:
            raise ValueError(f"Unknown driving actions: {sorted(unknown_actions)!r}")
        for action, binding in self.bindings.items():
            if self.backend == GAMEPAD_BACKEND and binding.is_raw_joystick:
                raise ValueError(
                    f"Gamepad action {action!r} must use an SDL3 semantic control"
                )
            if self.backend == GAMEPAD_BACKEND and binding.device != 0:
                raise ValueError("SDL3 gamepad bindings cannot select a device index")
            if self.backend == JOYSTICK_BACKEND and not binding.is_raw_joystick:
                raise ValueError(
                    f"Joystick action {action!r} must use an SDL3 joystick control"
                )
            if self.backend == JOYSTICK_BACKEND and binding.device >= len(self.devices):
                raise ValueError(
                    f"Joystick action {action!r} refers to missing device "
                    f"{binding.device}"
                )
        if not 0.0 < self.steering_range <= 1.0:
            raise ValueError("steering_range must be in (0, 1]")
        if not 0.0 <= self.steering_deadzone < 1.0:
            raise ValueError("steering_deadzone must be in [0, 1)")
        if not 0.0 <= self.ffb_gain <= 1.0:
            raise ValueError("ffb_gain must be in [0, 1]")
        if self.backend == GAMEPAD_BACKEND and self.ffb_enabled:
            raise ValueError("SDL3 gamepad profiles cannot enable wheel force feedback")

    @property
    def is_joystick(self) -> bool:
        return self.backend == JOYSTICK_BACKEND

    def binding(self, action: str) -> Binding | None:
        return self.bindings.get(action)

    def map_button(self, name: str) -> str:
        if self.swap_face_buttons:
            return _SWAPPED_FACE_BUTTONS.get(name, name)
        return name


@dataclass
class ControllerState:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed_mps: float = 0.0
    connected: bool = False
    reverse: bool = False
    axes: dict[str, float] = field(default_factory=dict)
    buttons: frozenset[str] = frozenset()
    device_names: tuple[str, ...] = ()
    input_kind: str = "gamepad"

    def copy(self) -> ControllerState:
        return replace(self, axes=dict(self.axes), buttons=frozenset(self.buttons))


def joystick_control_key(device: int, control: str) -> str:
    """Return the internal state key for one raw joystick control."""
    return f"d{device}:{control}"


def parse_joystick_control_key(value: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"d(\d+):(axis_\d+|button_\d+)", value)
    return None if match is None else (int(match.group(1)), match.group(2))


def default_controller_profile() -> ControllerProfile:
    return ControllerProfile(
        name="sdl3-default",
        display_name="SDL3 game controller",
        bindings={
            action: Binding(kind, control)
            for action, (kind, control) in _DEFAULT_BINDINGS.items()
        },
        invert_steering=True,
        is_default=True,
    )


def default_joystick_profile() -> ControllerProfile:
    """Return an empty generic-wheel profile used while configuring devices."""
    return ControllerProfile(
        name="sdl3-wheel",
        display_name="SDL3 wheel and pedals",
        backend=JOYSTICK_BACKEND,
        steering_range=1.0,
        steering_deadzone=0.01,
    )


def apply_steering_curve(
    value: float, *, deadzone: float = 0.0, scale: float = 1.0
) -> float:
    deadzone = max(0.0, min(0.95, float(deadzone)))
    if abs(value) <= deadzone:
        value = 0.0
    elif deadzone:
        value = (1.0 if value > 0.0 else -1.0) * (
            (abs(value) - deadzone) / (1.0 - deadzone)
        )
    return max(-1.0, min(1.0, value * float(scale)))


def normalize_pedal(value: float, binding: Binding) -> float:
    """Normalize a raw SDL joystick pedal from its captured resting value."""
    value = max(-1.0, min(1.0, float(value)))
    rest = binding.rest
    if rest is None:
        normalized = (value + 1.0) * 0.5
        return 1.0 - normalized if binding.invert else normalized
    if binding.invert:
        span = max(1e-6, rest + 1.0)
        return max(0.0, min(1.0, (rest - value) / span))
    span = max(1e-6, 1.0 - rest)
    return max(0.0, min(1.0, (value - rest) / span))


def user_controller_profiles_dir() -> Path:
    from omnidreams.scenes import FLASHDREAMS_CACHE_DIR

    return FLASHDREAMS_CACHE_DIR / "interactive-drive" / "controllers"


def profile_filename(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return f"{slug or 'profile'}.yaml"


def _binding_from_data(value: Any) -> Binding:
    if not isinstance(value, dict):
        raise TypeError(f"SDL3 binding must be a mapping, got {value!r}")
    return Binding(
        str(value["type"]),
        str(value["control"]),
        device=int(value.get("device", 0)),
        rest=None if value.get("rest") is None else float(value["rest"]),
        invert=bool(value.get("invert", False)),
    )


def _device_from_data(value: Any) -> DeviceSpec:
    if not isinstance(value, dict):
        raise TypeError(f"SDL3 joystick device must be a mapping, got {value!r}")
    return DeviceSpec(
        name=str(value["name"]),
        guid=str(value.get("guid") or ""),
        vendor_id=int(value.get("vendor_id", 0)),
        product_id=int(value.get("product_id", 0)),
        kind=str(value.get("kind") or "unknown"),
        ordinal=int(value.get("ordinal", 0)),
    )


def controller_profile_from_yaml_dict(data: dict[str, Any]) -> ControllerProfile:
    if data.get("schema_version") != 1:
        raise ValueError("SDL3 controller profile schema_version must be 1")
    raw_bindings = data.get("bindings")
    if not isinstance(raw_bindings, dict):
        raise TypeError("SDL3 controller profile bindings must be a mapping")
    backend = str(data["backend"])
    unknown_actions = set(map(str, raw_bindings)) - set(DRIVE_ACTIONS)
    if unknown_actions:
        raise ValueError(f"Unknown driving actions: {sorted(unknown_actions)!r}")
    bindings = {
        str(action): _binding_from_data(value) for action, value in raw_bindings.items()
    }
    raw_devices = data.get("devices", [])
    if not isinstance(raw_devices, list):
        raise TypeError("SDL3 controller profile devices must be a list")
    ffb = data.get("ffb") or {}
    if not isinstance(ffb, dict):
        raise TypeError("SDL3 controller profile ffb must be a mapping")
    return ControllerProfile(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        bindings=bindings,
        backend=backend,
        devices=tuple(_device_from_data(item) for item in raw_devices),
        swap_face_buttons=bool(data.get("swap_face_buttons", False)),
        invert_steering=bool(data.get("invert_steering", False)),
        steering_range=float(data.get("steering_range", 0.75)),
        steering_deadzone=float(data.get("steering_deadzone", 0.08)),
        ffb_enabled=bool(ffb.get("enabled", False)),
        ffb_mode=str(ffb.get("mode", "auto")),
        ffb_gain=float(ffb.get("gain", 0.5)),
        is_default=bool(data.get("is_default", False)),
    )


def _binding_to_data(binding: Binding) -> dict[str, Any]:
    data: dict[str, Any] = {"type": binding.kind, "control": binding.control}
    if binding.is_raw_joystick:
        data["device"] = binding.device
        if binding.rest is not None:
            data["rest"] = round(float(binding.rest), 6)
        if binding.invert:
            data["invert"] = True
    return data


def controller_profile_to_yaml_dict(profile: ControllerProfile) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "backend": profile.backend,
        "name": profile.name,
        "display_name": profile.display_name,
        "is_default": profile.is_default,
        "bindings": {
            action: _binding_to_data(binding)
            for action, binding in profile.bindings.items()
        },
        "swap_face_buttons": profile.swap_face_buttons,
        "invert_steering": profile.invert_steering,
        "steering_range": profile.steering_range,
        "steering_deadzone": profile.steering_deadzone,
    }
    if profile.is_joystick:
        data["devices"] = [
            {
                "name": device.name,
                "guid": device.guid,
                "vendor_id": device.vendor_id,
                "product_id": device.product_id,
                "kind": device.kind,
                "ordinal": device.ordinal,
            }
            for device in profile.devices
        ]
        data["ffb"] = {
            "enabled": profile.ffb_enabled,
            "mode": profile.ffb_mode,
            "gain": profile.ffb_gain,
        }
    return data


def load_controller_profile_files(
    directory: Path,
) -> tuple[tuple[Path, ControllerProfile], ...]:
    if not directory.is_dir():
        return ()
    loaded: list[tuple[Path, ControllerProfile]] = []
    for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise TypeError("profile root must be a mapping")
            loaded.append((path, controller_profile_from_yaml_dict(data)))
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            logger.warning(f"Skipping invalid input profile {path}: {exc}")
    return tuple(loaded)


def load_controller_profiles(directory: Path) -> tuple[ControllerProfile, ...]:
    return tuple(profile for _path, profile in load_controller_profile_files(directory))


def save_controller_profile(profile: ControllerProfile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / profile_filename(profile.name)
    update_profile_file(path, profile)
    return path


def update_profile_file(path: Path, profile: ControllerProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(controller_profile_to_yaml_dict(profile), sort_keys=False),
        encoding="utf-8",
    )


def delete_profile_file(path: Path) -> None:
    path.unlink(missing_ok=True)


class Sdl3ControllerBridge:
    """Translate SlangPy/SDL3 gamepad callbacks into drive commands."""

    def __init__(
        self,
        *,
        profile: ControllerProfile,
        control: Any | None = None,
        on_input: Callable[[ControllerState], None] | None = None,
    ) -> None:
        self._profile = profile
        self._control = control
        self._on_input = on_input
        self._window: Any | None = None
        self._button_enum: Any | None = None
        self._state = ControllerState(axes={axis: 0.0 for axis in SDL3_AXES})
        self._last_update_s = time.monotonic()
        self._last_button_down: str | None = None
        self._signed_triggers = False

    @property
    def profile(self) -> ControllerProfile:
        return self._profile

    @profile.setter
    def profile(self, value: ControllerProfile) -> None:
        if value.swap_face_buttons != self._profile.swap_face_buttons:
            raw_buttons = {
                self._profile.map_button(name) for name in self._state.buttons
            }
            self._state.buttons = frozenset(
                value.map_button(name) for name in raw_buttons
            )
            self._last_button_down = None
        self._profile = value
        self._publish()

    @property
    def state(self) -> ControllerState:
        return self._state.copy()

    @property
    def last_button_down(self) -> str | None:
        return self._last_button_down

    def take_last_button_down(self) -> str | None:
        value = self._last_button_down
        self._last_button_down = None
        return value

    def attach(self, window: Any, gamepad_button_enum: Any) -> None:
        self._window = window
        self._button_enum = gamepad_button_enum
        window.on_gamepad_event = self._on_gamepad_event
        window.on_gamepad_state = self._on_gamepad_state

    def start(self) -> None:
        """Start input handling; SDL3 gamepad callbacks begin on attachment."""

    def poll(self) -> None:
        """Allow uniform bridge polling; SlangPy pushes gamepad state callbacks."""

    def stop(self) -> None:
        if self._window is not None:
            self._window.on_gamepad_event = None
            self._window.on_gamepad_state = None
        self._window = None
        self._state.connected = False
        self._state.buttons = frozenset()
        if self._control is not None:
            release = getattr(self._control, "release_all", None)
            if release is not None:
                release()

    def _on_gamepad_event(self, event: Any) -> None:
        if event.is_connect():
            self._state.connected = True
        elif event.is_disconnect():
            self._state.connected = False
            self._state.buttons = frozenset()
            self._signed_triggers = False
            if self._control is not None:
                release = getattr(self._control, "release_all", None)
                if release is not None:
                    release()
        elif event.is_button_down() or event.is_button_up():
            raw_name = getattr(event.button, "name", str(event.button)).lower()
            name = self._profile.map_button(raw_name)
            buttons = set(self._state.buttons)
            if event.is_button_down():
                buttons.add(name)
                self._last_button_down = name
                self._handle_action_button(name)
            else:
                buttons.discard(name)
            self._state.buttons = frozenset(buttons)
            self._state.connected = True
        self._publish()

    def _on_gamepad_state(self, state: Any) -> None:
        axes = {
            axis: max(-1.0, min(1.0, float(getattr(state, axis, 0.0))))
            for axis in SDL3_AXES
        }
        if min(axes["left_trigger"], axes["right_trigger"]) <= -0.5:
            self._signed_triggers = True
        if self._signed_triggers:
            for axis in ("left_trigger", "right_trigger"):
                axes[axis] = (axes[axis] + 1.0) * 0.5
        self._state.axes = axes
        if self._button_enum is not None:
            self._state.buttons = frozenset(
                self._profile.map_button(raw_name)
                for raw_name in SDL3_BUTTONS
                if (button := getattr(self._button_enum, raw_name, None)) is not None
                and state.is_button_down(button)
            )
        self._state.connected = True
        self._publish()

    def _handle_action_button(self, name: str) -> None:
        if self._matches("reverse", name):
            self._state.reverse = not self._state.reverse
        elif self._matches("reset", name) and self._control is not None:
            callback = getattr(self._control, "request_reset", None)
            if callback is not None:
                callback()
        elif self._matches("exit", name) and self._control is not None:
            callback = getattr(self._control, "request_exit_scene", None)
            if callback is not None:
                callback()

    def _matches(self, action: str, name: str) -> bool:
        binding = self._profile.binding(action)
        return (
            binding is not None and binding.kind == "button" and binding.control == name
        )

    def _value(self, action: str) -> float:
        binding = self._profile.binding(action)
        if binding is None:
            return 0.0
        if binding.kind == "button":
            return 1.0 if binding.control in self._state.buttons else 0.0
        value = self._state.axes.get(binding.control, 0.0)
        if "trigger" in binding.control:
            return max(0.0, min(1.0, value))
        return max(-1.0, min(1.0, value))

    def _publish(self) -> None:
        steering = self._value("steering")
        if self._profile.invert_steering:
            steering = -steering
        steering = apply_steering_curve(
            steering,
            deadzone=self._profile.steering_deadzone,
            scale=self._profile.steering_range,
        )
        throttle = max(0.0, self._value("throttle"))
        brake = max(0.0, self._value("brake"))
        self._state.steering = steering
        self._state.throttle = throttle
        self._state.brake = brake
        self._state.target_speed_mps = self._update_target_speed(throttle, brake)
        if self._control is not None and self._state.connected:
            self._control.set_drive(
                steer=steering,
                throttle=throttle,
                brake=brake,
                reverse=self._state.reverse,
            )
        if self._on_input is not None:
            self._on_input(self.state)

    def _update_target_speed(self, throttle: float, brake: float) -> float:
        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self._last_update_s))
        self._last_update_s = now
        speed = self._state.target_speed_mps
        direction = -1.0 if self._state.reverse else 1.0
        if throttle > 0.01 and brake <= 0.05:
            speed += direction * 2.0 * throttle * dt
        elif brake > 0.01:
            delta = 12.0 * brake * dt
            speed = max(0.0, speed - delta) if speed >= 0 else min(0.0, speed + delta)
        elif self._state.reverse:
            speed = min(0.0, speed + 0.5 * dt)
        elif speed < 4.47:
            speed += (4.47 - speed) * 0.18 * dt
        else:
            speed = max(0.0, speed - 0.5 * dt)
        return max(-36.0, min(36.0, speed))


def create_input_bridge(
    *,
    profile: ControllerProfile,
    control: Any | None = None,
    on_input: Callable[[ControllerState], None] | None = None,
):
    """Create the appropriate SDL3 bridge for a saved profile."""
    if profile.is_joystick:
        from omnidreams.interactive_drive.input.sdl3_joystick import (
            Sdl3JoystickBridge,
        )

        return Sdl3JoystickBridge(profile=profile, control=control, on_input=on_input)
    return Sdl3ControllerBridge(profile=profile, control=control, on_input=on_input)
