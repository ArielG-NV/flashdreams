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

"""Portable SDL3 raw-joystick input for wheels and separate pedal sets."""

from __future__ import annotations

import ctypes
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from omnidreams.interactive_drive.input.wheel_profiles import (
    ControllerState,
    DeviceSpec,
    FFB_MODES,
    WheelProfile,
    apply_steering_curve,
    joystick_control_key,
    normalize_pedal,
)

_SCAN_INTERVAL_S = 0.5
"""Minimum interval between SDL3 hot-plug scans."""

_JOYSTICK_TYPE_NAMES = (
    "unknown",
    "gamepad",
    "wheel",
    "arcade-stick",
    "flight-stick",
    "dance-pad",
    "guitar",
    "drum-kit",
    "arcade-pad",
    "throttle",
)


def _load_sdl3():
    """Import PySDL3 after selecting a persistent, user-writable binary cache."""
    from omnidreams.scenes import FLASHDREAMS_CACHE_DIR

    binary_dir = FLASHDREAMS_CACHE_DIR / "interactive-drive" / "sdl3"
    binary_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SDL_BINARY_PATH", str(binary_dir))
    os.environ.setdefault("SDL_CHECK_VERSION", "0")
    os.environ.setdefault("SDL_LOG_LEVEL", "1")
    try:
        import sdl3
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "Generic wheel/pedal input requires PySDL3 from the "
            "interactive-drive extra. Re-run `uv sync --package "
            "flashdreams-omnidreams --extra interactive-drive`. The first "
            "use also downloads SDL3's native library into the FlashDreams cache."
        ) from exc
    return sdl3


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else "SDL3 joystick"


@dataclass(frozen=True)
class JoystickDevice:
    """One currently connected SDL3 joystick."""

    instance_id: int
    """SDL3 instance identifier, valid until the device disconnects."""

    name: str
    """Human-readable SDL3 device name."""

    guid: str
    """Platform-specific SDL3 device GUID."""

    vendor_id: int
    """USB vendor identifier, or zero when SDL3 cannot report one."""

    product_id: int
    """USB product identifier, or zero when SDL3 cannot report one."""

    kind: str
    """Normalized SDL3 joystick type name."""

    axis_count: int
    """Number of raw axes exposed by SDL3."""

    button_count: int
    """Number of raw buttons exposed by SDL3."""


@dataclass(frozen=True)
class JoystickSnapshot:
    """One polled raw state, normalized independently of the host platform."""

    device: JoystickDevice
    """Device associated with the sampled state."""

    axes: tuple[float, ...]
    """Raw axes normalized to the inclusive ``[-1, 1]`` range."""

    buttons: frozenset[int]
    """Indices of buttons held at sampling time."""


@dataclass
class _HapticState:
    """Open SDL3 haptic handle and its active centering effect."""

    handle: Any
    features: int
    mode: str | None = None
    effect_id: int | None = None
    constant_level: int | None = None
    autocenter_strength: int | None = None
    smoothed_strength: float = 0.0
    autocenter_failed: bool = False


class Sdl3JoystickSystem:
    """Own the PySDL3 joystick handles and hot-plug polling lifecycle."""

    def __init__(self, sdl: Any | None = None) -> None:
        self._sdl = sdl
        self._handles: dict[int, Any] = {}
        self._devices: dict[int, JoystickDevice] = {}
        self._haptics: dict[int, _HapticState] = {}
        self._started = False
        self._haptic_started = False
        self._last_scan_s = 0.0

    @property
    def devices(self) -> tuple[JoystickDevice, ...]:
        return tuple(self._devices.values())

    def start(self) -> None:
        if self._started:
            return
        if self._sdl is None:
            self._sdl = _load_sdl3()
        if not self._sdl.SDL_InitSubSystem(self._sdl.SDL_INIT_JOYSTICK):
            error = _decode(self._sdl.SDL_GetError())
            raise RuntimeError(f"SDL3 could not initialize joystick input: {error}")
        self._haptic_started = bool(
            self._sdl.SDL_InitSubSystem(self._sdl.SDL_INIT_HAPTIC)
        )
        self._started = True
        self._scan(force=True)

    def poll(self) -> tuple[JoystickSnapshot, ...]:
        if not self._started:
            self.start()
        self._sdl.SDL_UpdateJoysticks()
        self._scan()
        snapshots: list[JoystickSnapshot] = []
        for instance_id, device in tuple(self._devices.items()):
            handle = self._handles[instance_id]
            if not self._sdl.SDL_JoystickConnected(handle):
                continue
            axes = tuple(
                self._normalize_axis(self._sdl.SDL_GetJoystickAxis(handle, index))
                for index in range(device.axis_count)
            )
            buttons = frozenset(
                index
                for index in range(device.button_count)
                if self._sdl.SDL_GetJoystickButton(handle, index)
            )
            snapshots.append(JoystickSnapshot(device, axes, buttons))
        return tuple(snapshots)

    def set_ffb(
        self,
        instance_id: int,
        mode: str,
        gain: float,
        steering: float,
        *,
        speed_mps: float = 0.0,
        test_force: float | None = None,
    ) -> str | None:
        """Apply the requested SDL3 centering strategy and return the one used."""
        if mode not in FFB_MODES:
            raise ValueError(f"Unknown SDL3 force-feedback mode: {mode!r}")
        state = self._open_haptic(instance_id)
        if state is None:
            return None
        for candidate in self._ffb_candidates(mode, state.features):
            if mode == "auto" and candidate == "autocenter" and state.autocenter_failed:
                continue
            if candidate != state.mode:
                self._disable_haptic_state(state)
            if candidate == "autocenter":
                if test_force is None:
                    target = (
                        0.15
                        if abs(speed_mps) < 0.1
                        else 0.35 + 0.65 * min(1.0, abs(speed_mps) / 14.0)
                    )
                    state.smoothed_strength += 0.12 * (target - state.smoothed_strength)
                    fraction = state.smoothed_strength
                else:
                    fraction = 1.0
                strength = round(fraction * max(0.0, min(1.0, float(gain))) * 100.0)
                if state.mode == candidate and state.autocenter_strength == strength:
                    return candidate
                if self._sdl.SDL_SetHapticAutocenter(state.handle, strength):
                    state.mode = candidate
                    state.autocenter_strength = strength
                    return candidate
                if mode == "auto":
                    state.autocenter_failed = True
            elif self._set_constant_force(
                state,
                gain,
                steering,
                speed_mps=speed_mps,
                test_force=test_force,
            ):
                state.mode = candidate
                return candidate
        return None

    def disable_ffb(self, instance_id: int) -> None:
        """Stop any SDL3 centering effect active for one joystick."""
        state = self._haptics.get(instance_id)
        if state is not None:
            self._disable_haptic_state(state)

    def _open_haptic(self, instance_id: int) -> _HapticState | None:
        if not self._haptic_started:
            return None
        handle = self._handles.get(instance_id)
        if handle is None:
            return None
        state = self._haptics.get(instance_id)
        if state is not None:
            return state
        haptic = self._sdl.SDL_OpenHapticFromJoystick(handle)
        if not haptic:
            return None
        state = _HapticState(
            handle=haptic,
            features=int(self._sdl.SDL_GetHapticFeatures(haptic)),
        )
        self._haptics[instance_id] = state
        return state

    def _ffb_candidates(self, mode: str, features: int) -> tuple[str, ...]:
        autocenter = bool(features & int(self._sdl.SDL_HAPTIC_AUTOCENTER))
        constant_force = bool(features & int(self._sdl.SDL_HAPTIC_CONSTANT))
        supported = {
            "autocenter": autocenter,
            "constant_force": constant_force,
        }
        candidates = ("autocenter", "constant_force") if mode == "auto" else (mode,)
        return tuple(candidate for candidate in candidates if supported[candidate])

    def _set_constant_force(
        self,
        state: _HapticState,
        gain: float,
        steering: float,
        *,
        speed_mps: float,
        test_force: float | None,
    ) -> bool:
        gain = max(0.0, min(1.0, float(gain)))
        if test_force is None:
            target = (
                0.0
                if abs(speed_mps) < 0.1
                else 0.25 + 0.75 * min(1.0, abs(speed_mps) / 13.9)
            )
            state.smoothed_strength += 0.15 * (target - state.smoothed_strength)
            displacement = max(-1.0, min(1.0, float(steering)))
            shaped = math.copysign(math.sqrt(abs(displacement)), displacement)
            force = shaped * state.smoothed_strength
        else:
            force = max(-1.0, min(1.0, float(test_force)))
        level = round(force * gain * 32767.0)
        if state.effect_id is not None:
            if (
                state.constant_level is not None
                and abs(level - state.constant_level) <= 100
            ):
                return True
            effect = self._constant_effect(level)
            if not self._sdl.SDL_UpdateHapticEffect(
                state.handle, state.effect_id, ctypes.byref(effect)
            ):
                return False
            state.constant_level = level
            return True

        effect = self._constant_effect(level)
        effect_id = int(
            self._sdl.SDL_CreateHapticEffect(state.handle, ctypes.byref(effect))
        )
        if effect_id == int(self._sdl.SDL_HAPTIC_INFINITY):
            return False
        if not self._sdl.SDL_RunHapticEffect(state.handle, effect_id, 1):
            self._sdl.SDL_DestroyHapticEffect(state.handle, effect_id)
            return False
        state.effect_id = effect_id
        state.constant_level = level
        return True

    def _constant_effect(self, level: int) -> Any:
        effect = self._sdl.SDL_HapticEffect()
        effect.constant.type = self._sdl.SDL_HAPTIC_CONSTANT
        effect.constant.direction.type = self._sdl.SDL_HAPTIC_STEERING_AXIS
        effect.constant.direction.dir[0] = 0
        effect.constant.length = self._sdl.SDL_HAPTIC_INFINITY
        effect.constant.level = int(level)
        return effect

    def _disable_haptic_state(self, state: _HapticState) -> None:
        if state.mode == "autocenter":
            self._sdl.SDL_SetHapticAutocenter(state.handle, 0)
        if state.effect_id is not None:
            self._sdl.SDL_StopHapticEffect(state.handle, state.effect_id)
            self._sdl.SDL_DestroyHapticEffect(state.handle, state.effect_id)
        state.mode = None
        state.effect_id = None
        state.constant_level = None
        state.autocenter_strength = None
        state.smoothed_strength = 0.0

    def _close_haptic(self, instance_id: int) -> None:
        state = self._haptics.pop(instance_id, None)
        if state is None:
            return
        self._disable_haptic_state(state)
        self._sdl.SDL_CloseHaptic(state.handle)

    def close(self) -> None:
        if not self._started:
            return
        for instance_id in tuple(self._haptics):
            self._close_haptic(instance_id)
        for handle in self._handles.values():
            self._sdl.SDL_CloseJoystick(handle)
        self._handles.clear()
        self._devices.clear()
        if self._haptic_started:
            self._sdl.SDL_QuitSubSystem(self._sdl.SDL_INIT_HAPTIC)
        self._sdl.SDL_QuitSubSystem(self._sdl.SDL_INIT_JOYSTICK)
        self._started = False
        self._haptic_started = False

    def _scan(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_scan_s < _SCAN_INTERVAL_S:
            return
        self._last_scan_s = now
        count = ctypes.c_int()
        ids = self._sdl.SDL_GetJoysticks(ctypes.byref(count))
        try:
            connected_ids = (
                {int(ids[index]) for index in range(count.value)} if ids else set()
            )
        finally:
            if ids:
                self._sdl.SDL_free(ids)

        for instance_id in tuple(self._handles):
            if instance_id not in connected_ids:
                self._close_device(instance_id)
        for instance_id in sorted(connected_ids):
            if instance_id not in self._handles:
                self._open_device(instance_id)

    def _open_device(self, instance_id: int) -> None:
        handle = self._sdl.SDL_OpenJoystick(instance_id)
        if not handle:
            logger.warning(
                "SDL3 could not open joystick {}: {}",
                instance_id,
                _decode(self._sdl.SDL_GetError()),
            )
            return
        guid_buffer = ctypes.create_string_buffer(33)
        self._sdl.SDL_GUIDToString(
            self._sdl.SDL_GetJoystickGUIDForID(instance_id), guid_buffer, 33
        )
        raw_kind = int(self._sdl.SDL_GetJoystickTypeForID(instance_id))
        kind = (
            _JOYSTICK_TYPE_NAMES[raw_kind]
            if 0 <= raw_kind < len(_JOYSTICK_TYPE_NAMES)
            else "unknown"
        )
        self._handles[instance_id] = handle
        self._devices[instance_id] = JoystickDevice(
            instance_id=instance_id,
            name=_decode(self._sdl.SDL_GetJoystickNameForID(instance_id)),
            guid=guid_buffer.value.decode("ascii", errors="replace"),
            vendor_id=int(self._sdl.SDL_GetJoystickVendorForID(instance_id)),
            product_id=int(self._sdl.SDL_GetJoystickProductForID(instance_id)),
            kind=kind,
            axis_count=max(0, int(self._sdl.SDL_GetNumJoystickAxes(handle))),
            button_count=max(0, int(self._sdl.SDL_GetNumJoystickButtons(handle))),
        )

    def _close_device(self, instance_id: int) -> None:
        self._close_haptic(instance_id)
        handle = self._handles.pop(instance_id, None)
        if handle:
            self._sdl.SDL_CloseJoystick(handle)
        self._devices.pop(instance_id, None)

    @staticmethod
    def _normalize_axis(value: int) -> float:
        value = int(value)
        return value / (32767.0 if value >= 0 else 32768.0)


def _match_device(
    spec: DeviceSpec, snapshots: tuple[JoystickSnapshot, ...], used: set[int]
) -> JoystickSnapshot | None:
    matchers = (
        lambda device: (
            bool(spec.guid) and device.guid.casefold() == spec.guid.casefold()
        ),
        lambda device: (
            bool(spec.vendor_id and spec.product_id)
            and device.vendor_id == spec.vendor_id
            and device.product_id == spec.product_id
        ),
        lambda device: (
            bool(spec.name) and device.name.casefold() == spec.name.casefold()
        ),
    )
    for matches in matchers:
        candidates = [item for item in snapshots if matches(item.device)]
        if candidates:
            preferred = candidates[min(spec.ordinal, len(candidates) - 1)]
            if preferred.device.instance_id not in used:
                return preferred
            return next(
                (item for item in candidates if item.device.instance_id not in used),
                None,
            )
    return None


class Sdl3JoystickBridge:
    """Map one or more SDL3 joysticks to driving controls."""

    def __init__(
        self,
        *,
        profile: WheelProfile,
        control: Any | None = None,
        on_input: Callable[[ControllerState], None] | None = None,
        system: Sdl3JoystickSystem | Any | None = None,
    ) -> None:
        self._profile = profile
        self._control = control
        self._on_input = on_input
        self._system = system or Sdl3JoystickSystem()
        self._state = ControllerState(input_kind="joystick")
        self._last_update_s = time.monotonic()
        self._last_button_down: str | None = None
        self._previous_buttons: frozenset[str] = frozenset()
        self._logical_snapshots: tuple[JoystickSnapshot | None, ...] = ()
        self._ffb_signature: tuple[int, bool, str, float] | None = None
        self._ffb_test_force: float | None = None
        self._started = False

    @property
    def profile(self) -> WheelProfile:
        return self._profile

    @profile.setter
    def profile(self, value: WheelProfile) -> None:
        self._profile = value
        self._publish()

    @property
    def state(self) -> ControllerState:
        return self._state.copy()

    @property
    def available_devices(self) -> tuple[JoystickDevice, ...]:
        return self._system.devices

    @property
    def connected_device_indices(self) -> frozenset[int]:
        return frozenset(
            index
            for index, snapshot in enumerate(self._logical_snapshots)
            if snapshot is not None
        )

    @property
    def last_button_down(self) -> str | None:
        return self._last_button_down

    def take_last_button_down(self) -> str | None:
        value = self._last_button_down
        self._last_button_down = None
        return value

    def set_ffb_test_force(self, fraction: float | None) -> None:
        """Override runtime centering with a configurator motor-test force."""
        self._ffb_test_force = (
            None if fraction is None else max(-1.0, min(1.0, float(fraction)))
        )

    def attach(self, _window: Any, _gamepad_button_enum: Any) -> None:
        """Keep the same presenter interface; raw SDL3 input is polled."""

    def start(self) -> None:
        self._system.start()
        self._started = True

    def poll(self) -> None:
        if not self._started:
            self.start()
        snapshots = self._system.poll()
        self._logical_snapshots = self._resolve_snapshots(snapshots)
        axes: dict[str, float] = {}
        buttons: set[str] = set()
        device_names: list[str] = []
        for device_index, snapshot in enumerate(self._logical_snapshots):
            if snapshot is None:
                continue
            device_names.append(snapshot.device.name)
            axes.update(
                {
                    joystick_control_key(device_index, f"axis_{axis_index}"): value
                    for axis_index, value in enumerate(snapshot.axes)
                }
            )
            buttons.update(
                joystick_control_key(device_index, f"button_{button_index}")
                for button_index in snapshot.buttons
            )
        pressed = buttons - self._previous_buttons
        self._previous_buttons = frozenset(buttons)
        self._state.axes = axes
        self._state.buttons = frozenset(buttons)
        self._state.device_names = tuple(device_names)
        was_connected = self._state.connected
        self._state.connected = self._required_device_connected()
        if was_connected and not self._state.connected and self._control is not None:
            release = getattr(self._control, "release_all", None)
            if release is not None:
                release()
        for name in sorted(pressed):
            self._last_button_down = name
            self._handle_action_button(name)
        self._publish()
        self._apply_ffb()

    def stop(self) -> None:
        if self._started:
            self._disable_previous_ffb()
            self._system.close()
        self._started = False
        self._state.connected = False
        self._state.axes = {}
        self._state.buttons = frozenset()
        self._previous_buttons = frozenset()
        self._logical_snapshots = ()
        self._ffb_test_force = None
        if self._control is not None:
            release = getattr(self._control, "release_all", None)
            if release is not None:
                release()

    def device_specs(self) -> tuple[DeviceSpec, ...]:
        """Describe devices in the logical order used by raw bindings."""
        specs: list[DeviceSpec] = []
        seen: dict[tuple[str, int, int], int] = {}
        for index, snapshot in enumerate(self._logical_snapshots):
            if snapshot is None:
                if index < len(self._profile.devices):
                    specs.append(self._profile.devices[index])
                continue
            device = snapshot.device
            fingerprint = (device.name.casefold(), device.vendor_id, device.product_id)
            ordinal = seen.get(fingerprint, 0)
            seen[fingerprint] = ordinal + 1
            specs.append(
                DeviceSpec(
                    name=device.name,
                    guid=device.guid,
                    vendor_id=device.vendor_id,
                    product_id=device.product_id,
                    kind=device.kind,
                    ordinal=ordinal,
                )
            )
        return tuple(specs)

    def _resolve_snapshots(
        self, snapshots: tuple[JoystickSnapshot, ...]
    ) -> tuple[JoystickSnapshot | None, ...]:
        if not self._profile.devices:
            return tuple(snapshots)
        used: set[int] = set()
        resolved: list[JoystickSnapshot | None] = []
        for spec in self._profile.devices:
            match = _match_device(spec, snapshots, used)
            resolved.append(match)
            if match is not None:
                used.add(match.device.instance_id)
        return tuple(resolved)

    def _required_device_connected(self) -> bool:
        steering = self._profile.binding("steering")
        if steering is None:
            return any(item is not None for item in self._logical_snapshots)
        return (
            steering.device < len(self._logical_snapshots)
            and self._logical_snapshots[steering.device] is not None
        )

    def _matches(self, action: str, name: str) -> bool:
        binding = self._profile.binding(action)
        if binding is None or binding.kind != "button":
            return False
        return joystick_control_key(binding.device, binding.control) == name

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

    def _value(self, action: str) -> float:
        binding = self._profile.binding(action)
        if binding is None:
            return 0.0
        key = joystick_control_key(binding.device, binding.control)
        if binding.kind == "button":
            return 1.0 if key in self._state.buttons else 0.0
        if key not in self._state.axes:
            return 0.0
        value = self._state.axes[key]
        if action in {"throttle", "brake"}:
            return normalize_pedal(value, binding)
        return max(-1.0, min(1.0, value))

    def _apply_ffb(self) -> None:
        steering = self._profile.binding("steering")
        if steering is None or steering.device >= len(self._logical_snapshots):
            self._disable_previous_ffb()
            return
        snapshot = self._logical_snapshots[steering.device]
        if snapshot is None:
            self._disable_previous_ffb()
            return
        signature = (
            snapshot.device.instance_id,
            self._profile.ffb_enabled,
            self._profile.ffb_mode,
            self._profile.ffb_gain,
        )
        if (
            self._ffb_signature is not None
            and self._ffb_signature[1]
            and self._ffb_signature[0] != snapshot.device.instance_id
        ):
            self._system.disable_ffb(self._ffb_signature[0])
        if not self._profile.ffb_enabled:
            if self._ffb_signature is not None and self._ffb_signature[1]:
                self._system.disable_ffb(self._ffb_signature[0])
            self._ffb_signature = signature
            return
        resolved = self._system.set_ffb(
            snapshot.device.instance_id,
            self._profile.ffb_mode,
            self._profile.ffb_gain,
            self._value("steering"),
            speed_mps=self._state.target_speed_mps,
            test_force=self._ffb_test_force,
        )
        if resolved is None and signature != self._ffb_signature:
            logger.warning(
                "SDL3 {} force feedback is unavailable for {!r}; input remains active.",
                self._profile.ffb_mode,
                snapshot.device.name,
            )
        self._ffb_signature = signature

    def _disable_previous_ffb(self) -> None:
        if self._ffb_signature is not None and self._ffb_signature[1]:
            self._system.disable_ffb(self._ffb_signature[0])
        self._ffb_signature = None

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
