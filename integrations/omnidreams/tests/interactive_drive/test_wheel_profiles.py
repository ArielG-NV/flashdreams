# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SDL3 semantic-controller and generic-joystick profiles and bridges."""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
import yaml
from omnidreams.interactive_drive.input.input_guides import profile_guide
from omnidreams.interactive_drive.input.sdl3_joystick import (
    JoystickDevice,
    JoystickSnapshot,
    Sdl3JoystickBridge,
    Sdl3JoystickSystem,
)
from omnidreams.interactive_drive.input.wheel_profiles import (
    GAMEPAD_BACKEND,
    JOYSTICK_BACKEND,
    Binding,
    DeviceSpec,
    Sdl3ControllerBridge,
    WheelProfile,
    apply_steering_curve,
    default_controller_profile,
    load_wheel_profiles,
    normalize_pedal,
    save_wheel_profile,
    user_wheel_profiles_dir,
    wheel_profile_to_yaml_dict,
)

pytestmark = pytest.mark.ci_cpu


def _profile(**overrides) -> WheelProfile:
    values = {
        "name": "switch-pro",
        "display_name": "Switch Pro",
        "bindings": {
            "steering": Binding("axis", "left_x"),
            "throttle": Binding("axis", "right_trigger"),
            "brake": Binding("axis", "left_trigger"),
            "reverse": Binding("button", "b"),
            "reset": Binding("button", "start"),
            "exit": Binding("button", "back"),
        },
        "steering_range": 0.7,
        "steering_deadzone": 0.1,
        "is_default": True,
    }
    values.update(overrides)
    return WheelProfile(**values)


def test_default_mapping_uses_sdl3_semantic_controls() -> None:
    profile = default_controller_profile()
    assert profile.binding("steering") == Binding("axis", "left_x")
    assert profile.binding("throttle") == Binding("axis", "right_trigger")
    assert profile.binding("brake") == Binding("axis", "left_trigger")
    assert profile.binding("exit") == Binding("button", "back")
    assert profile.invert_steering is True


def test_profile_round_trip(tmp_path: Path) -> None:
    profile = _profile(swap_face_buttons=True)
    save_wheel_profile(profile, tmp_path)
    assert load_wheel_profiles(tmp_path) == (profile,)


def test_yaml_schema_contains_no_platform_device_identifiers() -> None:
    data = wheel_profile_to_yaml_dict(_profile())
    assert data["schema_version"] == 3
    assert data["backend"] == GAMEPAD_BACKEND
    assert data["bindings"]["throttle"] == {
        "type": "axis",
        "control": "right_trigger",
    }
    text = yaml.safe_dump(data)
    assert "device" not in text
    assert "code" not in text


def test_legacy_raw_profile_migrates_to_sdl3_joystick_indices(tmp_path: Path) -> None:
    legacy = {
        "name": "old-switch-pro",
        "display_name": "Nintendo Switch Pro Controller",
        "axis_map": {"steering": {"device": 0, "code": 0}},
        "throttle_buttons": [{"device": 0, "code": 313}],
        "brake_buttons": [{"device": 0, "code": 312}],
        "reset_buttons": [{"device": 0, "code": 315}],
        "ffb": {"enabled": True, "mode": "constant_force", "gain": 0.4},
    }
    (tmp_path / "legacy.yaml").write_text(yaml.safe_dump(legacy), encoding="utf-8")
    (profile,) = load_wheel_profiles(tmp_path)
    assert profile.backend == JOYSTICK_BACKEND
    assert profile.binding("steering") == Binding("axis", "axis_0")
    assert profile.binding("throttle") == Binding("button", "button_313")
    assert profile.binding("brake") == Binding("button", "button_312")
    assert profile.binding("reset") == Binding("button", "button_315")
    assert profile.ffb_enabled is True
    assert profile.ffb_mode == "constant_force"
    assert profile.ffb_gain == 0.4


def test_multi_device_wheel_profile_round_trip(tmp_path: Path) -> None:
    profile = WheelProfile(
        name="wheel-and-pedals",
        display_name="Wheel + Pedals",
        backend=JOYSTICK_BACKEND,
        devices=(
            DeviceSpec("Generic Wheel", guid="wheel-guid", kind="wheel"),
            DeviceSpec("USB Pedals", guid="pedal-guid", kind="unknown"),
        ),
        bindings={
            "steering": Binding("axis", "axis_0", device=0),
            "throttle": Binding("axis", "axis_1", device=1, rest=1.0, invert=True),
            "brake": Binding("axis", "axis_2", device=1, rest=1.0, invert=True),
            "reset": Binding("button", "button_4", device=0),
        },
        ffb_enabled=True,
        ffb_mode="constant_force",
        ffb_gain=0.65,
    )
    save_wheel_profile(profile, tmp_path)
    assert load_wheel_profiles(tmp_path) == (profile,)
    assert ("Generic Wheel: Axis 0", "Steer") in profile_guide(profile)
    assert ("USB Pedals: Axis 1", "Throttle") in profile_guide(profile)


def test_pedal_normalization_uses_captured_rest_and_direction() -> None:
    inverted = Binding("axis", "axis_0", rest=1.0, invert=True)
    assert normalize_pedal(1.0, inverted) == 0.0
    assert normalize_pedal(0.0, inverted) == 0.5
    assert normalize_pedal(-1.0, inverted) == 1.0


def test_invalid_semantic_control_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown SDL3 axis"):
        Binding("axis", "evdev-0x05")


def test_invalid_force_feedback_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="force-feedback mode"):
        _profile(ffb_mode="fanatec-special")


def test_auto_ffb_falls_back_to_constant_force_without_autocenter() -> None:
    class _HapticFeatures:
        SDL_HAPTIC_CONSTANT = 1
        SDL_HAPTIC_AUTOCENTER = 1 << 17

    system = Sdl3JoystickSystem(sdl=_HapticFeatures())
    assert system._ffb_candidates("auto", _HapticFeatures.SDL_HAPTIC_CONSTANT) == (
        "constant_force",
    )
    assert system._ffb_candidates(
        "auto",
        _HapticFeatures.SDL_HAPTIC_CONSTANT | _HapticFeatures.SDL_HAPTIC_AUTOCENTER,
    ) == ("autocenter", "constant_force")


def test_constant_force_tracks_steering_and_stops_cleanly() -> None:
    class _Direction(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("dir", ctypes.c_int * 3)]

    class _Constant(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("direction", _Direction),
            ("length", ctypes.c_uint32),
            ("level", ctypes.c_int16),
        ]

    class _Effect(ctypes.Union):
        _fields_ = [("type", ctypes.c_int), ("constant", _Constant)]

    class _ConstantOnlySdl:
        SDL_HAPTIC_CONSTANT = 1
        SDL_HAPTIC_AUTOCENTER = 1 << 17
        SDL_HAPTIC_STEERING_AXIS = 3
        SDL_HAPTIC_INFINITY = 0xFFFFFFFF
        SDL_HapticEffect = _Effect

        def __init__(self) -> None:
            self.created: list[int] = []
            self.updated: list[int] = []
            self.stopped: list[int] = []
            self.destroyed: list[int] = []
            self.autocenter_attempts: list[int] = []

        def SDL_OpenHapticFromJoystick(self, _joystick):
            return object()

        def SDL_GetHapticFeatures(self, _haptic) -> int:
            return self.SDL_HAPTIC_CONSTANT | self.SDL_HAPTIC_AUTOCENTER

        def SDL_CreateHapticEffect(self, _haptic, effect) -> int:
            self.created.append(effect._obj.constant.level)
            return 7

        def SDL_RunHapticEffect(self, _haptic, _effect_id, _iterations) -> bool:
            return True

        def SDL_UpdateHapticEffect(self, _haptic, _effect_id, effect) -> bool:
            self.updated.append(effect._obj.constant.level)
            return True

        def SDL_StopHapticEffect(self, _haptic, effect_id) -> bool:
            self.stopped.append(effect_id)
            return True

        def SDL_DestroyHapticEffect(self, _haptic, effect_id) -> None:
            self.destroyed.append(effect_id)

        def SDL_SetHapticAutocenter(self, _haptic, strength) -> bool:
            self.autocenter_attempts.append(strength)
            return False

    sdl = _ConstantOnlySdl()
    system = Sdl3JoystickSystem(sdl=sdl)
    system._haptic_started = True
    system._handles[42] = object()

    assert system.set_ffb(42, "auto", 0.5, 0.25, test_force=0.25) == "constant_force"
    assert sdl.created == [4096]
    assert system.set_ffb(42, "auto", 0.5, -0.5, test_force=-0.5) == "constant_force"
    assert sdl.updated == [-8192]
    assert sdl.autocenter_attempts == [50]

    system.disable_ffb(42)
    assert sdl.stopped == [7]
    assert sdl.destroyed == [7]


def test_apply_steering_curve() -> None:
    assert apply_steering_curve(0.05, deadzone=0.1) == 0.0
    assert apply_steering_curve(0.55, deadzone=0.1) == pytest.approx(0.5)
    assert apply_steering_curve(-1.0, scale=0.5) == -0.5
    assert apply_steering_curve(1.0, scale=2.0) == 1.0


class _Window:
    on_gamepad_event = None
    on_gamepad_state = None


class _Button:
    def __init__(self, name: str) -> None:
        self.name = name


class _ButtonEnum:
    pass


for _name in (
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
):
    setattr(_ButtonEnum, _name, _Button(_name))


class _State:
    left_x = 0.5
    left_y = 0.0
    right_x = 0.0
    right_y = 0.0
    left_trigger = -0.5
    right_trigger = 1.0

    def __init__(self, buttons=(), **axes) -> None:
        self._buttons = set(buttons)
        for name, value in axes.items():
            setattr(self, name, value)

    def is_button_down(self, button) -> bool:
        return button.name in self._buttons


class _Event:
    def __init__(self, kind: str, button: str = "a") -> None:
        self.kind = kind
        self.button = _Button(button)

    def is_connect(self) -> bool:
        return self.kind == "connect"

    def is_disconnect(self) -> bool:
        return self.kind == "disconnect"

    def is_button_down(self) -> bool:
        return self.kind == "down"

    def is_button_up(self) -> bool:
        return self.kind == "up"


class _Control:
    def __init__(self) -> None:
        self.command = None
        self.resets = 0
        self.exits = 0
        self.released = 0

    def set_drive(self, **command) -> None:
        self.command = command

    def request_reset(self) -> None:
        self.resets += 1

    def request_exit_scene(self) -> None:
        self.exits += 1

    def release_all(self) -> None:
        self.released += 1


def test_bridge_uses_window_sdl3_callbacks_for_switch_pro_triggers() -> None:
    window = _Window()
    control = _Control()
    bridge = Sdl3ControllerBridge(profile=_profile(), control=control)
    bridge.attach(window, _ButtonEnum)

    window.on_gamepad_event(_Event("connect"))
    window.on_gamepad_state(_State())

    assert bridge.state.connected is True
    assert bridge.state.steering == pytest.approx((0.5 - 0.1) / 0.9 * 0.7)
    assert control.command["throttle"] == 1.0
    assert control.command["brake"] == 0.25


def test_xbox_360_xinput_triggers_remain_independent_analog_axes() -> None:
    """Verify signed SDL trigger axes normalize independently to the drive range."""
    window = _Window()
    control = _Control()
    bridge = Sdl3ControllerBridge(profile=default_controller_profile(), control=control)
    bridge.attach(window, _ButtonEnum)

    window.on_gamepad_state(_State(left_trigger=-1.0, right_trigger=-1.0))
    assert bridge.state.brake == 0.0
    assert bridge.state.throttle == 0.0

    window.on_gamepad_state(_State(left_trigger=-0.2, right_trigger=0.5))

    assert control.command["steer"] < 0.0
    assert control.command["brake"] == pytest.approx(0.4)
    assert control.command["throttle"] == pytest.approx(0.75)
    assert bridge.state.axes["left_trigger"] == pytest.approx(0.4)
    assert bridge.state.axes["right_trigger"] == pytest.approx(0.75)


def test_zero_based_controller_triggers_are_not_rescaled() -> None:
    window = _Window()
    control = _Control()
    bridge = Sdl3ControllerBridge(profile=default_controller_profile(), control=control)
    bridge.attach(window, _ButtonEnum)

    window.on_gamepad_state(_State(left_trigger=0.25, right_trigger=0.75))

    assert control.command["brake"] == pytest.approx(0.25)
    assert control.command["throttle"] == pytest.approx(0.75)


def test_xinput_switch_label_mode_swaps_face_buttons_before_actions() -> None:
    window = _Window()
    control = _Control()
    profile = _profile(
        swap_face_buttons=True,
        bindings={
            **{
                action: binding
                for action, binding in _profile().bindings.items()
                if action != "reverse"
            },
            "reset": Binding("button", "b"),
        },
    )
    bridge = Sdl3ControllerBridge(profile=profile, control=control)
    bridge.attach(window, _ButtonEnum)

    window.on_gamepad_event(_Event("down", "a"))
    window.on_gamepad_state(_State(buttons={"a"}))

    assert control.resets == 1
    assert bridge.last_button_down == "b"
    assert bridge.state.buttons == frozenset({"b"})


def test_bridge_maps_action_buttons_and_hot_unplug() -> None:
    window = _Window()
    control = _Control()
    bridge = Sdl3ControllerBridge(profile=_profile(), control=control)
    bridge.attach(window, _ButtonEnum)

    window.on_gamepad_event(_Event("down", "b"))
    assert bridge.state.reverse is True
    window.on_gamepad_event(_Event("down", "start"))
    window.on_gamepad_event(_Event("down", "back"))
    assert control.resets == 1
    assert control.exits == 1

    window.on_gamepad_event(_Event("disconnect"))
    assert bridge.state.connected is False
    assert control.released == 1


class _JoystickSystem:
    def __init__(self, snapshots: tuple[JoystickSnapshot, ...]) -> None:
        self.snapshots = snapshots
        self.started = False
        self.closed = False
        self.ffb: list[tuple[int, str, float, float]] = []
        self.ffb_disabled: list[int] = []

    @property
    def devices(self):
        return tuple(snapshot.device for snapshot in self.snapshots)

    def start(self) -> None:
        self.started = True

    def poll(self) -> tuple[JoystickSnapshot, ...]:
        return self.snapshots

    def set_ffb(
        self,
        instance_id: int,
        mode: str,
        gain: float,
        steering: float,
        *,
        speed_mps: float = 0.0,
        test_force: float | None = None,
    ) -> str:
        self.ffb.append((instance_id, mode, gain, steering))
        return "constant_force" if mode == "auto" else mode

    def disable_ffb(self, instance_id: int) -> None:
        self.ffb_disabled.append(instance_id)

    def close(self) -> None:
        self.closed = True


def _joystick_device(instance_id: int, name: str, guid: str) -> JoystickDevice:
    return JoystickDevice(
        instance_id=instance_id,
        name=name,
        guid=guid,
        vendor_id=0x1234,
        product_id=instance_id,
        kind="wheel" if "Wheel" in name else "unknown",
        axis_count=3,
        button_count=8,
    )


def test_generic_wheel_and_separate_pedals_drive_through_sdl3() -> None:
    wheel = _joystick_device(10, "Generic Wheel", "wheel-guid")
    pedals = _joystick_device(20, "USB Pedals", "pedal-guid")
    system = _JoystickSystem(
        (
            JoystickSnapshot(wheel, (0.5, 0.0, 0.0), frozenset({4})),
            JoystickSnapshot(pedals, (-1.0, 0.0, 1.0), frozenset()),
        )
    )
    profile = WheelProfile(
        name="generic-wheel",
        display_name="Generic Wheel + USB Pedals",
        backend=JOYSTICK_BACKEND,
        devices=(
            DeviceSpec("Generic Wheel", guid="wheel-guid"),
            DeviceSpec("USB Pedals", guid="pedal-guid"),
        ),
        bindings={
            "steering": Binding("axis", "axis_0", device=0),
            "throttle": Binding("axis", "axis_0", device=1, rest=1.0, invert=True),
            "brake": Binding("axis", "axis_1", device=1, rest=1.0, invert=True),
            "reset": Binding("button", "button_4", device=0),
        },
        steering_deadzone=0.0,
        steering_range=1.0,
        ffb_enabled=True,
        ffb_gain=0.6,
    )
    control = _Control()
    bridge = Sdl3JoystickBridge(profile=profile, control=control, system=system)

    bridge.start()
    bridge.poll()

    assert bridge.state.connected is True
    assert bridge.state.device_names == ("Generic Wheel", "USB Pedals")
    assert control.command["steer"] == pytest.approx(0.5)
    assert control.command["throttle"] == pytest.approx(1.0)
    assert control.command["brake"] == pytest.approx(0.5)
    assert control.resets == 1
    assert system.ffb == [(10, "auto", 0.6, 0.5)]

    system.snapshots = ()
    bridge.poll()
    assert bridge.state.connected is False
    assert control.released == 1
    assert system.ffb_disabled[-1] == 10


def test_missing_separate_pedals_preserves_wheel_connection() -> None:
    wheel = _joystick_device(10, "Generic Wheel", "wheel-guid")
    system = _JoystickSystem((JoystickSnapshot(wheel, (-0.25, 0.0, 0.0), frozenset()),))
    profile = WheelProfile(
        name="generic-wheel",
        display_name="Generic Wheel + USB Pedals",
        backend=JOYSTICK_BACKEND,
        devices=(
            DeviceSpec("Generic Wheel", guid="wheel-guid"),
            DeviceSpec("USB Pedals", guid="pedal-guid"),
        ),
        bindings={
            "steering": Binding("axis", "axis_0", device=0),
            "throttle": Binding("axis", "axis_0", device=1, rest=1.0, invert=True),
        },
        steering_deadzone=0.0,
        steering_range=1.0,
    )
    bridge = Sdl3JoystickBridge(profile=profile, system=system)

    bridge.poll()

    assert bridge.state.connected is True
    assert bridge.state.steering == pytest.approx(-0.25)
    assert bridge.state.throttle == 0.0


def test_user_dir_follows_cache_env(monkeypatch, tmp_path) -> None:
    from omnidreams import scenes

    monkeypatch.setattr(scenes, "FLASHDREAMS_CACHE_DIR", tmp_path)
    assert user_wheel_profiles_dir() == tmp_path / "interactive-drive" / "wheels"
