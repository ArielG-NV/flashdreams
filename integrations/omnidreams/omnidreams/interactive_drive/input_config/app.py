# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tkinter wizard for SDL3 controller, wheel, and pedal configuration."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from omnidreams.interactive_drive.input.wheel_profiles import (
    FFB_MODES,
    GAMEPAD_BACKEND,
    JOYSTICK_BACKEND,
    SDL3_AXES,
    Binding,
    ControllerState,
    DeviceSpec,
    WheelProfile,
    apply_steering_curve,
    create_input_bridge,
    default_controller_profile,
    default_wheel_profile,
    delete_profile_file,
    joystick_control_key,
    load_wheel_profile_files,
    normalize_pedal,
    parse_joystick_control_key,
    profile_filename,
    save_wheel_profile,
    update_profile_file,
    user_wheel_profiles_dir,
    wheel_profile_to_yaml_dict,
)
from omnidreams.interactive_drive.input_config.capture import CaptureSession
from omnidreams.interactive_drive.log import configure_logging

try:  # Tkinter is stdlib but needs the system Tk package installed.
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # pragma: no cover - exercised only on Tk-less hosts
    tk = None
    ttk = None
    messagebox = None

_REFERENCE_SCREEN_WIDTH = 1920
"""Reference display width for resolution-aware UI scaling."""

_REFERENCE_SCREEN_HEIGHT = 1080
"""Reference display height for resolution-aware UI scaling."""

_REFERENCE_DPI = 96.0
"""Tk/Windows baseline display density in pixels per inch."""

_MAX_UI_SCALE = 3.0
"""Upper bound for incorrect display metadata."""

_CANVAS_W = 690
"""Logical width of the persistent live-input panel."""

_CANVAS_H = 164
"""Logical height of the persistent live-input panel."""

_WHEEL_MAX_DEG = 120.0
"""Maximum wheel rotation drawn in either direction."""

_TICK_MS = 60
"""UI and SDL3 polling period."""

_ACTION_LABELS = {
    "steering": "Steering",
    "throttle": "Throttle",
    "brake": "Brake",
    "reverse": "Reverse",
    "reset": "Reset / respawn",
    "exit": "Exit scene",
}


def _enable_high_dpi_awareness() -> None:
    """Enable per-monitor DPI awareness before Tk creates a Windows window."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        import ctypes

        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError):
        pass
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def _ui_scale_for_display(width: int, height: int, dpi: float) -> float:
    """Return a bounded scale derived from display resolution and density."""
    width = max(1, int(width))
    height = max(1, int(height))
    dpi_scale = max(1.0, float(dpi) / _REFERENCE_DPI)
    resolution_ratio = min(
        width / _REFERENCE_SCREEN_WIDTH,
        height / _REFERENCE_SCREEN_HEIGHT,
    )
    resolution_scale = max(1.0, math.sqrt(resolution_ratio))
    fit_scale = max(1.0, min(width / 840.0, height / 790.0))
    return min(_MAX_UI_SCALE, max(dpi_scale, resolution_scale), fit_scale)


def _display_scale(root) -> float:
    """Read monitor metrics from Tk and choose a UI scale."""
    try:
        dpi = float(root.winfo_fpixels("1i"))
    except (TypeError, ValueError, tk.TclError):
        dpi = _REFERENCE_DPI
    return _ui_scale_for_display(
        root.winfo_screenwidth(), root.winfo_screenheight(), dpi
    )


def _device_spec(device, ordinal: int) -> DeviceSpec:
    """Build a persistent profile device descriptor from an SDL3 device."""
    return DeviceSpec(
        name=device.name,
        guid=device.guid,
        vendor_id=device.vendor_id,
        product_id=device.product_id,
        kind=device.kind,
        ordinal=ordinal,
    )


class ConfigApp:
    """Wizard controller built around a single ``tk.Tk`` root."""

    def __init__(self, root, *, spy=None) -> None:
        if spy is None:
            try:
                import slangpy as spy
            except ImportError as exc:
                raise RuntimeError(
                    "Input configuration requires the interactive-drive extra: "
                    "uv sync --package flashdreams-omnidreams --extra interactive-drive"
                ) from exc

        self.root = root
        self.spy = spy
        self.ui_scale = _display_scale(root)
        self.root.tk.call("tk", "scaling", (_REFERENCE_DPI / 72.0) * self.ui_scale)
        self.root.title("interactive-drive input configuration")
        self.root.geometry(f"{self._px(780)}x{self._px(740)}")
        self.root.minsize(self._px(760), self._px(700))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.state: dict[str, Any] = {}
        self.device_type_var = tk.StringVar(value="wheel")
        self.activity_var = tk.StringVar(value="")
        self._saved = False
        self._step_index = 0
        self._editing: tuple[Path, WheelProfile] | None = None
        self._capture = CaptureSession()
        self._capture_buttons: dict[str, Any] = {}
        self._capture_results: dict[str, Any] = {}
        self._last_state = ControllerState(input_kind="joystick")
        self._bindings: dict[str, Binding] = {}
        self._available_devices: tuple[Any, ...] = ()
        self._available_specs: tuple[DeviceSpec, ...] = ()
        self._ffb_testing = False
        self._ffb_test_phase = 0.0
        self._closing = False

        self._event_window = spy.Window(
            width=1,
            height=1,
            title="interactive-drive SDL3 input host",
            mode=spy.WindowMode.minimized,
            resizable=False,
        )
        self._working_profile = replace(
            default_wheel_profile(), bindings={}, is_default=False
        )
        self._bridge = self._new_bridge(self._working_profile)

        self._build_chrome()
        self._render()
        self.root.after(0, self._tick)

    def _px(self, value: float) -> int:
        return max(1, round(float(value) * self.ui_scale))

    def _canvas_coords(self, *values: float) -> tuple[float, ...]:
        return tuple(float(value) * self.ui_scale for value in values)

    def _new_bridge(self, profile: WheelProfile):
        bridge_profile = (
            replace(profile, ffb_enabled=False) if profile.is_joystick else profile
        )
        bridge = create_input_bridge(profile=bridge_profile, on_input=self._on_input)
        bridge.attach(self._event_window, self.spy.GamepadButton)
        bridge.start()
        return bridge

    def _sync_bridge_profile(self) -> None:
        profile = self._working_profile
        self._bridge.profile = (
            replace(profile, ffb_enabled=False) if profile.is_joystick else profile
        )

    def _replace_bridge(self, profile: WheelProfile) -> None:
        self._bridge.stop()
        self._working_profile = profile
        self._bindings = dict(profile.bindings)
        self._last_state = ControllerState(
            input_kind="joystick" if profile.is_joystick else "gamepad"
        )
        self._bridge = self._new_bridge(profile)

    ## Window chrome

    def _build_chrome(self) -> None:
        footer = ttk.Frame(self.root, padding=(self._px(16), self._px(10)))
        footer.pack(side="bottom", fill="x")
        ttk.Button(footer, text="Cancel", command=self._on_close).pack(side="left")
        self.primary_btn = ttk.Button(footer, text="Next", command=self._on_primary)
        self.primary_btn.pack(side="right")
        self.back_btn = ttk.Button(footer, text="Back", command=self._on_back)
        self.back_btn.pack(side="right", padx=(0, self._px(8)))

        live = ttk.LabelFrame(
            self.root,
            text="Live inputs",
            padding=(self._px(8), self._px(4)),
        )
        live.pack(
            side="bottom",
            fill="x",
            padx=self._px(12),
            pady=(0, self._px(4)),
        )
        ttk.Label(live, textvariable=self.activity_var, foreground="#2f8f2f").pack(
            anchor="w"
        )
        self.live_canvas = tk.Canvas(
            live,
            width=self._px(_CANVAS_W),
            height=self._px(_CANVAS_H),
            highlightthickness=0,
        )
        self.live_canvas.pack(anchor="w")

        header = ttk.Frame(self.root, padding=(self._px(16), self._px(10)))
        header.pack(side="top", fill="x")
        self.title_var = tk.StringVar()
        ttk.Label(
            header,
            textvariable=self.title_var,
            font=("TkDefaultFont", 15, "bold"),
        ).pack(anchor="w")
        self.step_var = tk.StringVar()
        ttk.Label(header, textvariable=self.step_var, foreground="#888").pack(
            anchor="w"
        )

        self.content = ttk.Frame(self.root, padding=(self._px(16), self._px(4)))
        self.content.pack(side="top", fill="both", expand=True)

    def _clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()

    ## Step navigation

    def _steps(self) -> list[str]:
        steps = ["welcome", "device", "controls", "buttons"]
        if self.state.get("device_type", self.device_type_var.get()) == "wheel":
            steps.append("ffb")
        return [*steps, "details", "review"]

    def _current_step(self) -> str:
        steps = self._steps()
        return steps[min(self._step_index, len(steps) - 1)]

    def _render(self) -> None:
        self._stop_ffb_test()
        self._capture.cancel()
        self._capture_buttons = {}
        self._capture_results = {}
        self._clear_content()
        if self._editing is not None:
            self.step_var.set("Editing an existing profile")
            self.back_btn.state(["!disabled"])
            self.primary_btn.config(text="Save changes")
            self._build_edit()
            return
        step = self._current_step()
        steps = self._steps()
        self.step_var.set(f"Step {self._step_index + 1} of {len(steps)}")
        self.back_btn.state(["!disabled"] if self._step_index > 0 else ["disabled"])
        self.primary_btn.config(text="Save profile" if step == "review" else "Next")
        getattr(self, f"_build_{step}")()

    def _on_primary(self) -> None:
        if self._saved:
            self._on_close()
            return
        if self._editing is not None:
            self._save_edit()
            return
        step = self._current_step()
        ok, message = self._validate(step)
        if not ok:
            messagebox.showwarning("Not ready", message)
            return
        if step == "review":
            self._save()
            return
        self._step_index += 1
        self._render()

    def _on_back(self) -> None:
        if self._editing is not None:
            self._editing = None
            self._step_index = 0
            self._replace_bridge(
                replace(default_wheel_profile(), bindings={}, is_default=False)
            )
            self._render()
            return
        if self._step_index > 0:
            self._step_index -= 1
            self._render()

    ## Welcome and profile management

    def _build_welcome(self) -> None:
        self.title_var.set("Input device configuration")
        entries = load_wheel_profile_files(user_wheel_profiles_dir())
        saved = ttk.LabelFrame(
            self.content,
            text="Saved profiles",
            padding=(self._px(10), self._px(6)),
        )
        saved.pack(fill="x", pady=(0, self._px(10)))
        if not entries:
            ttk.Label(saved, text="No saved profiles yet.").pack(anchor="w")
        else:
            for path, profile in entries:
                row = ttk.Frame(saved)
                row.pack(fill="x", pady=self._px(2))
                tag = "  [default]" if profile.is_default else ""
                ttk.Label(
                    row,
                    text=f"{profile.display_name}{tag}",
                    width=30,
                    anchor="w",
                ).pack(side="left")
                ttk.Button(
                    row,
                    text="Edit",
                    width=6,
                    command=lambda p=path, pr=profile: self._start_edit(p, pr),
                ).pack(side="left", padx=self._px(2))
                ttk.Button(
                    row,
                    text="Unset default" if profile.is_default else "Make default",
                    width=13,
                    command=lambda p=path, pr=profile: self._toggle_default(p, pr),
                ).pack(side="left", padx=self._px(2))
                ttk.Button(
                    row,
                    text="Delete",
                    width=7,
                    command=lambda p=path, pr=profile: self._delete_profile(p, pr),
                ).pack(side="left", padx=self._px(2))

        ttk.Label(
            self.content,
            text="Create a new profile",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(self._px(6), self._px(2)))
        ttk.Label(
            self.content,
            wraplength=self._px(680),
            justify="left",
            text="Pick the device type, then click Next to detect and calibrate it.",
        ).pack(anchor="w", pady=(0, self._px(6)))
        ttk.Radiobutton(
            self.content,
            text="Steering wheel + pedals",
            value="wheel",
            variable=self.device_type_var,
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.content,
            text="Game controller / gamepad (stick + triggers)",
            value="controller",
            variable=self.device_type_var,
        ).pack(anchor="w")

    def _start_edit(self, path: Path, profile: WheelProfile) -> None:
        self._editing = (path, profile)
        self._replace_bridge(profile)
        self._render()

    def _delete_profile(self, path: Path, profile: WheelProfile) -> None:
        if messagebox.askyesno("Delete profile", f"Delete '{profile.display_name}'?"):
            delete_profile_file(path)
            self._editing = None
            self._render()

    def _toggle_default(self, path: Path, profile: WheelProfile) -> None:
        make_default = not profile.is_default
        for other_path, other in load_wheel_profile_files(user_wheel_profiles_dir()):
            desired = (
                make_default
                if other_path == path
                else (False if make_default else other.is_default)
            )
            if desired != other.is_default:
                update_profile_file(other_path, replace(other, is_default=desired))
        self._render()

    def _build_edit(self) -> None:
        _path, profile = self._editing
        self.title_var.set(f"Edit: {profile.display_name}")
        self._edit_display_name = tk.StringVar(value=profile.display_name)
        self._edit_invert_steer = tk.BooleanVar(value=profile.invert_steering)
        self._edit_range = tk.DoubleVar(value=profile.steering_range)
        self._edit_deadzone = tk.DoubleVar(value=profile.steering_deadzone)
        self._edit_default = tk.BooleanVar(value=profile.is_default)
        self._edit_swap = tk.BooleanVar(value=profile.swap_face_buttons)
        self._edit_ffb = tk.BooleanVar(value=profile.ffb_enabled)
        self._edit_ffb_mode = tk.StringVar(value=profile.ffb_mode)
        self._edit_ffb_gain = tk.DoubleVar(value=profile.ffb_gain)

        status = (
            "Operate the controls to preview this profile live."
            if self._last_state.connected
            else "Connect the saved device to preview it live."
        )
        ttk.Label(
            self.content,
            foreground="#2f8f2f",
            wraplength=self._px(680),
            text=status,
        ).pack(anchor="w", pady=(0, self._px(6)))

        form = ttk.Frame(self.content)
        form.pack(fill="x")
        ttk.Label(form, text="Display name").grid(
            row=0, column=0, sticky="w", pady=self._px(4)
        )
        ttk.Entry(form, textvariable=self._edit_display_name, width=44).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(
            self.content,
            foreground="#666",
            wraplength=self._px(680),
            text="Bindings: " + self._profile_binding_summary(profile),
        ).pack(anchor="w", pady=(self._px(4), self._px(6)))
        if not profile.is_joystick:
            ttk.Checkbutton(
                self.content,
                text="Swap controller labels (Nintendo through XInput: A/B and X/Y)",
                variable=self._edit_swap,
                command=self._apply_edit_preview,
            ).pack(anchor="w", pady=(0, self._px(6)))
        ttk.Checkbutton(
            self.content,
            text="Invert steering",
            variable=self._edit_invert_steer,
            command=self._apply_edit_preview,
        ).pack(anchor="w")
        self._slider_row(
            "Steering range (sensitivity)",
            self._edit_range,
            0.1,
            1.0,
            self._apply_edit_preview,
        )
        self._slider_row(
            "Steering deadzone",
            self._edit_deadzone,
            0.0,
            0.3,
            self._apply_edit_preview,
        )
        if profile.is_joystick:
            ffb_row = ttk.Frame(self.content)
            ffb_row.pack(fill="x", pady=self._px(2))
            ttk.Checkbutton(
                ffb_row, text="Wheel centering", variable=self._edit_ffb
            ).pack(side="left")
            ttk.Label(ffb_row, text="Mode").pack(
                side="left", padx=(self._px(8), self._px(3))
            )
            ttk.Combobox(
                ffb_row,
                textvariable=self._edit_ffb_mode,
                values=FFB_MODES,
                state="readonly",
                width=15,
            ).pack(side="left")
            ttk.Scale(
                ffb_row,
                from_=0.0,
                to=1.0,
                length=self._px(220),
                variable=self._edit_ffb_gain,
            ).pack(side="left", padx=self._px(8))
            ttk.Button(ffb_row, text="Test", command=self._edit_ffb_test).pack(
                side="left"
            )
            ttk.Button(ffb_row, text="Stop", command=self._stop_ffb_test).pack(
                side="left", padx=self._px(4)
            )
            device_text = "\n".join(
                f"• {device.name} ({device.kind}, {device.guid or 'no GUID'})"
                for device in profile.devices
            )
            ttk.Label(
                self.content,
                text="Matched SDL3 devices:\n" + (device_text or "• none"),
                justify="left",
                wraplength=self._px(680),
            ).pack(anchor="w", pady=(self._px(8), 0))
        ttk.Checkbutton(
            self.content,
            text="Use as the default profile",
            variable=self._edit_default,
        ).pack(anchor="w", pady=(self._px(8), 0))
        ttk.Button(
            self.content,
            text="Delete this profile",
            command=lambda: self._delete_profile(*self._editing),
        ).pack(anchor="w", pady=(self._px(10), 0))

    def _profile_binding_summary(self, profile: WheelProfile) -> str:
        return ", ".join(
            f"{_ACTION_LABELS[action]}={self._binding_label(binding, profile)}"
            for action, binding in profile.bindings.items()
        )

    def _apply_edit_preview(self, *_args) -> None:
        if self._editing is None:
            return
        _path, profile = self._editing
        preview = replace(
            profile,
            invert_steering=bool(self._edit_invert_steer.get()),
            steering_range=float(self._edit_range.get()),
            steering_deadzone=float(self._edit_deadzone.get()),
            swap_face_buttons=bool(self._edit_swap.get()),
        )
        self._working_profile = preview
        self._sync_bridge_profile()

    def _save_edit(self) -> None:
        path, profile = self._editing
        updated = replace(
            profile,
            display_name=self._edit_display_name.get().strip() or profile.display_name,
            invert_steering=bool(self._edit_invert_steer.get()),
            steering_range=float(self._edit_range.get()),
            steering_deadzone=float(self._edit_deadzone.get()),
            swap_face_buttons=bool(self._edit_swap.get()),
            ffb_enabled=bool(self._edit_ffb.get()) if profile.is_joystick else False,
            ffb_mode=self._edit_ffb_mode.get() if profile.is_joystick else "auto",
            ffb_gain=float(self._edit_ffb_gain.get()),
            is_default=bool(self._edit_default.get()),
        )
        update_profile_file(path, updated)
        self._demote_other_defaults(path, updated)
        messagebox.showinfo("Saved", f"Updated {path.name}.")
        self._editing = None
        self._step_index = 0
        self._replace_bridge(
            replace(default_wheel_profile(), bindings={}, is_default=False)
        )
        self._render()

    ## Device selection

    def _build_device(self) -> None:
        if self._working_profile.is_joystick:
            self._build_wheel_device_selection()
        else:
            self._build_controller_connection()

    def _build_wheel_device_selection(self) -> None:
        self.title_var.set("Select your device(s)")
        ttk.Label(
            self.content,
            wraplength=self._px(680),
            justify="left",
            text=(
                "Select the wheel and every pedal device it uses. Ctrl+click to "
                "select more than one device when the wheel base and pedals are "
                "separate USB devices. Operate a control and confirm activity in "
                "the Live inputs panel, then click Next."
            ),
        ).pack(anchor="w", pady=(0, self._px(8)))
        row = ttk.Frame(self.content)
        row.pack(fill="both", expand=True)
        self.device_list = tk.Listbox(
            row, height=7, exportselection=False, selectmode="extended"
        )
        self.device_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(row, command=self.device_list.yview)
        scroll.pack(side="right", fill="y")
        self.device_list.config(yscrollcommand=scroll.set)
        self.device_list.bind("<<ListboxSelect>>", self._on_device_selected)
        ttk.Button(self.content, text="Rescan", command=self._refresh_devices).pack(
            anchor="w", pady=self._px(8)
        )
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        self._bridge.poll()
        self._available_devices = tuple(self._bridge.available_devices)
        seen: dict[tuple[str, int, int], int] = {}
        specs: list[DeviceSpec] = []
        for device in self._available_devices:
            fingerprint = (device.name.casefold(), device.vendor_id, device.product_id)
            ordinal = seen.get(fingerprint, 0)
            seen[fingerprint] = ordinal + 1
            specs.append(_device_spec(device, ordinal))
        self._available_specs = tuple(specs)
        self.device_list.delete(0, "end")
        selected = tuple(self.state.get("devices", ()))
        for index, device in enumerate(self._available_devices):
            self.device_list.insert(
                "end",
                f"{device.name}  [{device.kind}; {device.axis_count} axes; "
                f"{device.button_count} buttons; "
                f"{device.vendor_id:04x}:{device.product_id:04x}]",
            )
            if (
                index < len(self._available_specs)
                and self._available_specs[index] in selected
            ):
                self.device_list.selection_set(index)
        if not self._available_devices:
            self.device_list.insert("end", "(no SDL3 joystick devices found)")
        if selected:
            self._apply_device_selection(selected)

    def _on_device_selected(self, _event=None) -> None:
        selected = tuple(
            self._available_specs[index]
            for index in self.device_list.curselection()
            if index < len(self._available_specs)
        )
        self._apply_device_selection(selected)

    def _apply_device_selection(self, selected: tuple[DeviceSpec, ...]) -> None:
        previous = tuple(self.state.get("devices", ()))
        self.state["devices"] = selected
        if selected != previous:
            self._bindings = {}
            self.state.pop("_profile", None)
        self._working_profile = replace(
            self._working_profile,
            backend=JOYSTICK_BACKEND,
            devices=selected,
            bindings=dict(self._bindings),
        )
        self._sync_bridge_profile()

    def _build_controller_connection(self) -> None:
        self.title_var.set("Controller settings and connection")
        ttk.Label(
            self.content,
            wraplength=self._px(680),
            justify="left",
            text=(
                "Connect the controller over USB or Bluetooth, then operate a stick "
                "or button. SDL3 supplies one standardized Xbox-style layout for "
                "Switch Pro, Xbox/XInput, and PlayStation controllers."
            ),
        ).pack(anchor="w", pady=(0, self._px(10)))
        self.controller_swap_var = tk.BooleanVar(
            value=bool(self.state.get("swap_face_buttons", False))
        )
        ttk.Checkbutton(
            self.content,
            text="Swap controller labels (Nintendo through XInput: A/B and X/Y)",
            variable=self.controller_swap_var,
            command=self._apply_controller_swap,
        ).pack(anchor="w", pady=(0, self._px(10)))
        self.controller_status_var = tk.StringVar()
        ttk.Label(
            self.content,
            textvariable=self.controller_status_var,
            foreground="#2f8f2f",
        ).pack(anchor="w")
        self._update_controller_status()

    def _apply_controller_swap(self) -> None:
        enabled = bool(self.controller_swap_var.get())
        self.state["swap_face_buttons"] = enabled
        self._working_profile = replace(
            self._working_profile, swap_face_buttons=enabled
        )
        self._sync_bridge_profile()

    def _update_controller_status(self) -> None:
        if not hasattr(self, "controller_status_var"):
            return
        self.controller_status_var.set(
            "Controller connected — operate it to confirm below."
            if self._last_state.connected
            else "Waiting for an SDL3 game controller…"
        )

    ## Control and button capture

    def _build_controls(self) -> None:
        self.title_var.set("Calibrate controls")
        is_wheel = self._working_profile.is_joystick
        ttk.Label(
            self.content,
            wraplength=self._px(680),
            justify="left",
            text=(
                "For each control, click Start listening and move it a little. "
                "SDL3 captures the axis, device, released position, and pedal "
                "direction. The axis menu is a manual fallback."
            ),
        ).pack(anchor="w", pady=(0, self._px(8)))
        self._control_section(
            "steering", "Turn the wheel LEFT" if is_wheel else "Move the stick LEFT"
        )
        self._control_section(
            "throttle",
            "Press the throttle pedal" if is_wheel else "Press the throttle trigger",
        )
        self._control_section(
            "brake", "Press the brake pedal" if is_wheel else "Press the brake trigger"
        )

    def _control_section(self, action: str, instruction: str) -> None:
        frame = ttk.LabelFrame(
            self.content,
            text=_ACTION_LABELS[action],
            padding=(self._px(10), self._px(6)),
        )
        frame.pack(fill="x", pady=self._px(3))
        ttk.Label(frame, text=f"Click Start, then {instruction} a little.").pack(
            anchor="w"
        )
        row = ttk.Frame(frame)
        row.pack(anchor="w", fill="x", pady=self._px(4))
        button = ttk.Button(
            row,
            text="Start listening",
            command=lambda selected=action: self._start_capture(selected),
        )
        button.pack(side="left")
        self._capture_buttons[action] = button
        self._axis_override(row, action)
        result = tk.StringVar(value=self._binding_summary(action))
        self._capture_results[action] = result
        ttk.Label(
            frame,
            textvariable=result,
            foreground="#2f6fbf",
            wraplength=self._px(660),
        ).pack(anchor="w")
        if action == "steering":
            self.invert_steering_var = tk.BooleanVar(
                value=self._working_profile.invert_steering
            )
            ttk.Checkbutton(
                frame,
                text="Invert steering (toggle if left/right feels reversed)",
                variable=self.invert_steering_var,
                command=self._apply_steering_invert,
            ).pack(anchor="w")
        elif self._working_profile.is_joystick:
            binding = self._bindings.get(action)
            pedal_var = tk.BooleanVar(value=bool(binding and binding.invert))
            setattr(self, f"{action}_invert_var", pedal_var)
            ttk.Checkbutton(
                frame,
                text=f"Invert {action} travel",
                variable=pedal_var,
                command=lambda selected=action: self._apply_pedal_invert(selected),
            ).pack(anchor="w")

    def _axis_override(self, parent, action: str) -> None:
        ttk.Label(parent, text="  Axis:").pack(side="left")
        options: dict[str, Binding] = {}
        for key in sorted(self._last_state.axes):
            raw = parse_joystick_control_key(key)
            if raw is None:
                if key not in SDL3_AXES:
                    continue
                binding = Binding("axis", key)
            else:
                rest = (
                    self._last_state.axes[key]
                    if action in {"throttle", "brake"}
                    else None
                )
                binding = Binding("axis", raw[1], device=raw[0], rest=rest)
            options[self._binding_label(binding, self._working_profile)] = binding
        labels = list(options) or ["(none)"]
        current = self._bindings.get(action)
        current_label = (
            self._binding_label(current, self._working_profile)
            if current is not None
            else labels[0]
        )
        choice = tk.StringVar(
            value=current_label if current_label in labels else labels[0]
        )

        def on_pick(label: str) -> None:
            binding = options.get(label)
            if binding is not None:
                self._set_binding(action, binding)

        ttk.OptionMenu(parent, choice, choice.get(), *labels, command=on_pick).pack(
            side="left", padx=self._px(6)
        )

    def _start_capture(self, action: str) -> None:
        if not self._last_state.connected:
            messagebox.showwarning(
                "No device", "Connect and select the SDL3 input device first."
            )
            return
        if self._capture.listening and self._capture.action == action:
            self._capture.cancel()
            self._capture_buttons[action].config(text="Start listening")
            self._capture_results[action].set("Cancelled.")
            return
        for button in self._capture_buttons.values():
            button.config(text="Start listening")
        self._bridge.take_last_button_down()
        self._capture.start(action, self._bridge.state)
        self._capture_buttons[action].config(text="Listening... (click to cancel)")
        self._capture_results[action].set("Move or press the control to bind it.")

    def _on_input(self, state: ControllerState) -> None:
        self._last_state = state
        self._update_controller_status()
        if not self._capture.listening:
            return
        baseline = self._capture.baseline
        result = self._capture.feed(
            state, last_button_down=self._bridge.take_last_button_down()
        )
        if result is None:
            return
        action, binding = result
        if action == "steering" and binding.kind == "axis" and baseline is not None:
            key = self._binding_state_key(binding)
            invert = state.axes.get(key, 0.0) < baseline.axes.get(key, 0.0)
            self._working_profile = replace(
                self._working_profile, invert_steering=invert
            )
            if hasattr(self, "invert_steering_var"):
                self.invert_steering_var.set(invert)
        self._set_binding(action, binding)
        button = self._capture_buttons.get(action)
        if button is not None:
            button.config(text="Start listening")

    def _set_binding(self, action: str, binding: Binding) -> None:
        self._bindings[action] = binding
        self._working_profile = replace(
            self._working_profile, bindings=dict(self._bindings)
        )
        self._sync_bridge_profile()
        result = self._capture_results.get(action)
        if result is not None:
            result.set(self._binding_summary(action))
        if action in {"throttle", "brake"} and binding.is_raw_joystick:
            pedal_var = getattr(self, f"{action}_invert_var", None)
            if pedal_var is not None:
                pedal_var.set(binding.invert)

    def _binding_state_key(self, binding: Binding) -> str:
        return (
            joystick_control_key(binding.device, binding.control)
            if binding.is_raw_joystick
            else binding.control
        )

    def _binding_label(self, binding: Binding, profile: WheelProfile) -> str:
        control = binding.control.replace("_", " ").title()
        if binding.is_raw_joystick:
            device = (
                profile.devices[binding.device].name
                if binding.device < len(profile.devices)
                else f"Device {binding.device + 1}"
            )
            return f"{device}: {control}"
        return control

    def _binding_summary(self, action: str) -> str:
        binding = self._bindings.get(action)
        if binding is None:
            return "Not bound"
        detail = self._binding_label(binding, self._working_profile)
        if binding.kind == "axis" and action in {"throttle", "brake"}:
            direction = "inverted" if binding.invert else "normal"
            return f"{detail} ({direction} travel)"
        return detail

    def _apply_steering_invert(self) -> None:
        self._working_profile = replace(
            self._working_profile,
            invert_steering=bool(self.invert_steering_var.get()),
        )
        self._sync_bridge_profile()

    def _apply_pedal_invert(self, action: str) -> None:
        binding = self._bindings.get(action)
        if binding is None or binding.kind != "axis":
            return
        pedal_var = getattr(self, f"{action}_invert_var")
        self._set_binding(action, replace(binding, invert=bool(pedal_var.get())))

    def _build_buttons(self) -> None:
        self.title_var.set("Bind buttons (optional)")
        ttk.Label(
            self.content,
            wraplength=self._px(680),
            justify="left",
            text=(
                "Optionally bind reverse, reset / respawn, and exit scene. Click "
                "Bind, then press the button. Keyboard shortcuts remain available."
            ),
        ).pack(anchor="w", pady=(0, self._px(8)))
        for action in ("reverse", "reset", "exit"):
            frame = ttk.LabelFrame(
                self.content,
                text=_ACTION_LABELS[action],
                padding=(self._px(10), self._px(6)),
            )
            frame.pack(fill="x", pady=self._px(4))
            result = tk.StringVar(value=self._binding_summary(action))
            self._capture_results[action] = result
            row = ttk.Frame(frame)
            row.pack(anchor="w", fill="x")
            button = ttk.Button(
                row,
                text="Bind button",
                command=lambda selected=action: self._start_capture(selected),
            )
            button.pack(side="left")
            self._capture_buttons[action] = button
            ttk.Button(
                row,
                text="Clear",
                command=lambda selected=action: self._clear_binding(selected),
            ).pack(side="left", padx=self._px(8))
            ttk.Label(frame, textvariable=result, foreground="#2f6fbf").pack(
                anchor="w", pady=(self._px(4), 0)
            )

    def _clear_binding(self, action: str) -> None:
        self._bindings.pop(action, None)
        self._working_profile = replace(
            self._working_profile, bindings=dict(self._bindings)
        )
        self._sync_bridge_profile()
        self._capture.cancel()
        result = self._capture_results.get(action)
        if result is not None:
            result.set("Not bound")

    ## Force feedback and details

    def _build_ffb(self) -> None:
        self.title_var.set("Force feedback (optional)")
        self.ffb_enabled_var = tk.BooleanVar(
            value=bool(self.state.get("ffb_enabled", False))
        )
        self.ffb_mode_var = tk.StringVar(value=self.state.get("ffb_mode", "auto"))
        self.ffb_gain_var = tk.DoubleVar(value=float(self.state.get("ffb_gain", 0.6)))
        ttk.Label(
            self.content,
            wraplength=self._px(680),
            justify="left",
            text=(
                "SDL3 wheel centering makes a compatible wheel resist turning and "
                "return to center. Auto prefers hardware autocenter, then falls back "
                "to a constant-force effect for wheels such as Fanatec. Leave it off "
                "for devices without a motor."
            ),
        ).pack(anchor="w", pady=(0, self._px(10)))
        ttk.Checkbutton(
            self.content,
            text="Enable wheel centering",
            variable=self.ffb_enabled_var,
        ).pack(anchor="w")
        mode_row = ttk.Frame(self.content)
        mode_row.pack(anchor="w", pady=(self._px(8), 0))
        ttk.Label(mode_row, text="Mode").pack(side="left")
        ttk.Combobox(
            mode_row,
            textvariable=self.ffb_mode_var,
            values=FFB_MODES,
            state="readonly",
            width=18,
        ).pack(side="left", padx=self._px(8))
        gain_row = ttk.Frame(self.content)
        gain_row.pack(anchor="w", pady=self._px(8), fill="x")
        ttk.Label(gain_row, text="Gain").pack(side="left")
        ttk.Scale(
            gain_row,
            from_=0.0,
            to=1.0,
            length=self._px(300),
            variable=self.ffb_gain_var,
        ).pack(side="left", padx=self._px(8))
        test_row = ttk.Frame(self.content)
        test_row.pack(anchor="w", pady=self._px(4))
        ttk.Button(test_row, text="Test", command=self._ffb_test).pack(side="left")
        ttk.Button(test_row, text="Stop", command=self._stop_ffb_test).pack(
            side="left", padx=self._px(8)
        )

    def _ffb_test(self) -> None:
        gain = float(self.ffb_gain_var.get())
        mode = self.ffb_mode_var.get()
        self._ffb_testing = True
        self._ffb_test_phase = 0.0
        self._bridge.profile = replace(
            self._working_profile, ffb_enabled=True, ffb_mode=mode, ffb_gain=gain
        )
        self.activity_var.set(f"Testing SDL3 wheel {mode} at {gain:.0%}")

    def _edit_ffb_test(self) -> None:
        gain = float(self._edit_ffb_gain.get())
        mode = self._edit_ffb_mode.get()
        self._ffb_testing = True
        self._ffb_test_phase = 0.0
        self._bridge.profile = replace(
            self._working_profile, ffb_enabled=True, ffb_mode=mode, ffb_gain=gain
        )
        self.activity_var.set(f"Testing SDL3 wheel {mode} at {gain:.0%}")

    def _stop_ffb_test(self) -> None:
        if not self._ffb_testing:
            return
        self._ffb_testing = False
        self._bridge.set_ffb_test_force(None)
        self._bridge.profile = replace(self._working_profile, ffb_enabled=False)

    def _slider_row(
        self, label: str, var, low: float, high: float, callback=None
    ) -> None:
        row = ttk.Frame(self.content)
        row.pack(fill="x", pady=self._px(2))
        ttk.Label(row, text=label, width=26, anchor="w").pack(side="left")
        ttk.Scale(
            row,
            from_=low,
            to=high,
            length=self._px(220),
            variable=var,
        ).pack(side="left", padx=self._px(8))
        value_label = ttk.Label(row, width=5)
        value_label.pack(side="left")

        def update(*_args) -> None:
            value_label.config(text=f"{float(var.get()):.2f}")
            if callback is not None:
                callback()

        var.trace_add("write", update)
        update()

    def _build_details(self) -> None:
        self.title_var.set("Settings & detection")
        default_name = (
            self._working_profile.devices[0].name
            if self._working_profile.devices
            else "SDL3 game controller"
        )
        self.display_name_var = tk.StringVar(
            value=self.state.get("display_name", default_name)
        )
        self.profile_name_var = tk.StringVar(
            value=self.state.get("name", profile_filename(default_name)[:-5])
        )
        self.is_default_var = tk.BooleanVar(value=self.state.get("is_default", True))
        self.steering_range_var = tk.DoubleVar(
            value=self.state.get("steering_range", self._working_profile.steering_range)
        )
        self.steering_deadzone_var = tk.DoubleVar(
            value=self.state.get(
                "steering_deadzone", self._working_profile.steering_deadzone
            )
        )
        form = ttk.Frame(self.content)
        form.pack(fill="x")
        ttk.Label(form, text="Display name").grid(
            row=0, column=0, sticky="w", pady=self._px(4)
        )
        ttk.Entry(form, textvariable=self.display_name_var, width=46).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(form, text="Profile name").grid(
            row=1, column=0, sticky="w", pady=self._px(4)
        )
        ttk.Entry(form, textvariable=self.profile_name_var, width=46).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Label(
            self.content, text="Steering feel (turn the device to preview):"
        ).pack(anchor="w", pady=(self._px(8), 0))
        self._slider_row(
            "Steering range (sensitivity)",
            self.steering_range_var,
            0.1,
            1.0,
            self._apply_detail_preview,
        )
        self._slider_row(
            "Steering deadzone",
            self.steering_deadzone_var,
            0.0,
            0.3,
            self._apply_detail_preview,
        )
        if self._working_profile.is_joystick and self._working_profile.devices:
            ttk.Label(
                self.content,
                text="SDL3 matches: "
                + " + ".join(device.name for device in self._working_profile.devices),
                foreground="#555",
                wraplength=self._px(680),
            ).pack(anchor="w", pady=(self._px(8), 0))
        ttk.Checkbutton(
            self.content,
            text="Use as the default profile",
            variable=self.is_default_var,
        ).pack(anchor="w", pady=self._px(6))

    def _apply_detail_preview(self) -> None:
        if not hasattr(self, "steering_range_var"):
            return
        self._working_profile = replace(
            self._working_profile,
            steering_range=float(self.steering_range_var.get()),
            steering_deadzone=float(self.steering_deadzone_var.get()),
            swap_face_buttons=bool(self.state.get("swap_face_buttons", False)),
        )
        self._sync_bridge_profile()

    def _build_review(self) -> None:
        self.title_var.set("Review and save")
        profile = self._compose_profile()
        self.state["_profile"] = profile
        preview = yaml.safe_dump(
            wheel_profile_to_yaml_dict(profile),
            sort_keys=False,
            default_flow_style=False,
        )
        ttk.Label(
            self.content,
            text=(
                "Will be written to:\n"
                f"{user_wheel_profiles_dir() / profile_filename(profile.name)}"
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, self._px(8)))
        text = tk.Text(self.content, height=15, width=66)
        text.pack(fill="both", expand=True)
        text.insert("1.0", preview)
        text.config(state="disabled")

    ## Validation and save

    def _validate(self, step: str) -> tuple[bool, str]:
        if step == "welcome":
            kind = self.device_type_var.get()
            previous_swap = bool(self.state.get("swap_face_buttons", False))
            self.state = {"device_type": kind}
            profile = (
                default_wheel_profile()
                if kind == "wheel"
                else default_controller_profile()
            )
            if kind == "controller":
                self.state["swap_face_buttons"] = previous_swap
                profile = replace(profile, swap_face_buttons=previous_swap)
            self._replace_bridge(replace(profile, bindings={}, is_default=False))
            return True, ""
        if step == "device":
            if self._working_profile.is_joystick:
                devices = tuple(self.state.get("devices", ()))
                if not devices:
                    return False, "Select at least one wheel or pedal device."
                connected = self._bridge.connected_device_indices
                if connected != frozenset(range(len(devices))):
                    return False, "Reconnect every selected wheel/pedal device."
            elif not self._last_state.connected:
                return False, "Connect and operate the game controller first."
            return True, ""
        if step == "controls":
            missing = [
                _ACTION_LABELS[action]
                for action in ("steering", "throttle", "brake")
                if action not in self._bindings
            ]
            if missing:
                return False, "Calibrate these controls first: " + ", ".join(missing)
            return True, ""
        if step == "ffb":
            self._stop_ffb_test()
            self.state["ffb_enabled"] = bool(self.ffb_enabled_var.get())
            self.state["ffb_mode"] = self.ffb_mode_var.get()
            self.state["ffb_gain"] = float(self.ffb_gain_var.get())
            return True, ""
        if step == "details":
            name = self.profile_name_var.get().strip()
            if not name:
                return False, "Profile name cannot be empty."
            self.state.update(
                name=name,
                display_name=self.display_name_var.get().strip() or name,
                is_default=bool(self.is_default_var.get()),
                steering_range=float(self.steering_range_var.get()),
                steering_deadzone=float(self.steering_deadzone_var.get()),
            )
            return True, ""
        return True, ""

    def _compose_profile(self) -> WheelProfile:
        bindings = dict(self._bindings)
        devices = self._working_profile.devices
        if self._working_profile.is_joystick:
            used = sorted(
                {
                    binding.device
                    for binding in bindings.values()
                    if binding.is_raw_joystick
                }
            )
            remap = {old: new for new, old in enumerate(used)}
            devices = tuple(devices[index] for index in used)
            bindings = {
                action: replace(binding, device=remap[binding.device])
                if binding.is_raw_joystick
                else binding
                for action, binding in bindings.items()
            }
        return WheelProfile(
            name=self.state["name"],
            display_name=self.state["display_name"],
            bindings=bindings,
            backend=(
                JOYSTICK_BACKEND
                if self._working_profile.is_joystick
                else GAMEPAD_BACKEND
            ),
            devices=devices,
            swap_face_buttons=bool(self.state.get("swap_face_buttons", False)),
            invert_steering=self._working_profile.invert_steering,
            steering_range=float(self.state.get("steering_range", 1.0)),
            steering_deadzone=float(self.state.get("steering_deadzone", 0.0)),
            ffb_enabled=(
                bool(self.state.get("ffb_enabled", False))
                if self._working_profile.is_joystick
                else False
            ),
            ffb_mode=str(self.state.get("ffb_mode", "auto")),
            ffb_gain=float(self.state.get("ffb_gain", 0.5)),
            is_default=bool(self.state.get("is_default", True)),
        )

    def _save(self) -> None:
        profile = self.state.get("_profile") or self._compose_profile()
        try:
            path = save_wheel_profile(profile, user_wheel_profiles_dir())
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self._demote_other_defaults(path, profile)
        self._saved = True
        self.primary_btn.config(text="Close")
        self.back_btn.state(["disabled"])
        messagebox.showinfo(
            "Profile saved",
            f"Saved to:\n{path}\n\nLaunch the demo with:\n"
            "uv run --package flashdreams-omnidreams interactive-drive",
        )

    def _demote_other_defaults(self, path: Path, profile: WheelProfile) -> None:
        if not profile.is_default:
            return
        for other_path, other in load_wheel_profile_files(user_wheel_profiles_dir()):
            if other_path != path and other.is_default:
                update_profile_file(other_path, replace(other, is_default=False))

    ## Live panel

    def _tick(self) -> None:
        if self._closing:
            return
        if self._event_window.should_close():
            self._on_close()
            return
        self._event_window.process_events()
        if self._ffb_testing:
            self._ffb_test_phase += (_TICK_MS / 1000.0) * math.tau * 0.8
            self._bridge.set_ffb_test_force(math.sin(self._ffb_test_phase) * 0.55)
        self._bridge.poll()
        self._draw_live()
        self.root.after(_TICK_MS, self._tick)

    def _draw_live(self) -> None:
        canvas = self.live_canvas
        canvas.delete("all")
        editing = self._editing is not None
        step = self._current_step()
        if not editing and step in {"welcome", "review"}:
            self.activity_var.set("")
            return
        if self._capture.listening:
            self.activity_var.set("Listening — move or press the control…")
        elif self._last_state.connected:
            names = " + ".join(self._last_state.device_names)
            self.activity_var.set(
                f"Activity from {names}" if names else "SDL3 controller connected"
            )
        else:
            self.activity_var.set("Waiting for SDL3 input…")

        steer, throttle, brake = self._preview_values()
        self._draw_wheel(canvas, 78, 70, 56, steer)
        self._draw_pedal(canvas, 168, throttle, "Throttle", "#76b900")
        self._draw_pedal(canvas, 226, brake, "Brake", "#d05a5a")
        self._draw_axis_strip(canvas, 300, self._last_state.axes)
        held = sorted(self._last_state.buttons)
        canvas.create_text(
            10,
            _CANVAS_H - 6,
            anchor="w",
            fill="#666",
            font=("TkFixedFont", 8),
            text="Buttons held: " + (", ".join(held[:8]) if held else "(none)"),
        )
        canvas.scale("all", 0, 0, self.ui_scale, self.ui_scale)

    def _preview_values(self) -> tuple[float, float, float]:
        steer = self._binding_value("steering")
        if self._working_profile.invert_steering:
            steer = -steer
        steer = apply_steering_curve(
            steer,
            deadzone=self._working_profile.steering_deadzone,
            scale=self._working_profile.steering_range,
        )
        return (
            steer,
            max(0.0, self._binding_value("throttle")),
            max(0.0, self._binding_value("brake")),
        )

    def _binding_value(self, action: str) -> float:
        binding = self._bindings.get(action)
        if binding is None:
            return 0.0
        key = self._binding_state_key(binding)
        if binding.kind == "button":
            return 1.0 if key in self._last_state.buttons else 0.0
        if key not in self._last_state.axes:
            return 0.0
        value = self._last_state.axes[key]
        if action in {"throttle", "brake"} and binding.is_raw_joystick:
            return normalize_pedal(value, binding)
        if action in {"throttle", "brake"} and "trigger" in binding.control:
            return max(0.0, min(1.0, value))
        return max(-1.0, min(1.0, value))

    def _draw_wheel(self, canvas, cx: int, cy: int, radius: int, steer: float) -> None:
        canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#999",
            width=self._px(5),
        )
        angle = math.radians(-steer * _WHEEL_MAX_DEG)
        for spoke in range(3):
            spoke_angle = angle + spoke * (2.0 * math.pi / 3.0)
            x = cx + radius * math.sin(spoke_angle)
            y = cy - radius * math.cos(spoke_angle)
            canvas.create_line(
                cx,
                cy,
                x,
                y,
                fill="#76b900" if spoke == 0 else "#bbb",
                width=self._px(5 if spoke == 0 else 3),
            )
        canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#555", outline="")
        canvas.create_text(
            cx,
            cy + radius + 14,
            fill="#666",
            text=f"{round(steer * _WHEEL_MAX_DEG):+d}°",
        )

    def _draw_pedal(self, canvas, x: int, value: float, label: str, color: str) -> None:
        top, bottom, width = 16, 118, 34
        value = max(0.0, min(1.0, value))
        canvas.create_rectangle(
            x, top, x + width, bottom, outline="#999", width=self._px(1)
        )
        fill_height = value * (bottom - top)
        canvas.create_rectangle(
            x,
            bottom - fill_height,
            x + width,
            bottom,
            fill=color,
            outline="",
        )
        canvas.create_text(
            x + width / 2,
            top - 8,
            fill="#666",
            font=("TkDefaultFont", 8),
            text=f"{value:.0%}",
        )
        canvas.create_text(
            x + width / 2,
            bottom + 14,
            fill="#666",
            font=("TkDefaultFont", 8),
            text=label,
        )

    def _draw_axis_strip(self, canvas, x0: int, axes: dict[str, float]) -> None:
        bar_x, bar_width, row_height = x0 + 78, 130, 16
        canvas.create_text(
            x0,
            6,
            anchor="w",
            fill="#888",
            font=("TkDefaultFont", 8, "bold"),
            text="Axes",
        )
        for index, (name, value) in enumerate(sorted(axes.items())[:8]):
            fraction = (
                max(0.0, min(1.0, value))
                if "trigger" in name
                else max(0.0, min(1.0, (value + 1.0) * 0.5))
            )
            y = 20 + index * row_height
            short_name = name.replace("button_", "b").replace("axis_", "a")
            canvas.create_text(
                x0,
                y,
                anchor="w",
                fill="#666",
                font=("TkFixedFont", 8),
                text=short_name[:12],
            )
            canvas.create_rectangle(
                bar_x,
                y - 5,
                bar_x + bar_width,
                y + 5,
                outline="#aaa",
                width=self._px(1),
            )
            canvas.create_rectangle(
                bar_x,
                y - 5,
                bar_x + fraction * bar_width,
                y + 5,
                fill="#5a9bd5",
                outline="",
            )
            canvas.create_text(
                bar_x + bar_width + 6,
                y,
                anchor="w",
                fill="#666",
                font=("TkFixedFont", 8),
                text=f"{value:+.2f}",
            )

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_ffb_test()
        try:
            self._bridge.stop()
            self._event_window.close()
        finally:
            self.root.destroy()


def main() -> None:
    configure_logging()
    if tk is None:
        logger.error(
            "Tkinter is not available. Install your platform's Tk package and retry."
        )
        raise SystemExit(1)
    _enable_high_dpi_awareness()
    root = tk.Tk()
    try:
        ConfigApp(root)
    except RuntimeError as exc:
        root.destroy()
        logger.error(str(exc))
        raise SystemExit(1) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
