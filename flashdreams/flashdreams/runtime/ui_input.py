# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral raw input canonicalization for Dear ImGui."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import (
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

ImGuiInputKind = Literal[
    "mouse_position",
    "mouse_button",
    "mouse_wheel",
    "key",
    "text",
    "focus",
    "display_size",
]

POINTER_MOVE_EVENT_TYPE = "pointer_move"
"""Raw pointer-position event type shared by live input targets."""

POINTER_BUTTON_EVENT_TYPE = "pointer_button"
"""Raw pointer-button event type shared by live input targets."""

POINTER_WHEEL_EVENT_TYPE = "pointer_wheel"
"""Raw pointer-wheel event type shared by live input targets."""

TEXT_INPUT_EVENT_TYPE = "text_input"
"""Raw committed-text event type shared by live input targets."""

FOCUS_EVENT_TYPE = "focus"
"""Raw viewport-focus event type shared by live input targets."""

VIEWPORT_EVENT_TYPE = "viewport"
"""Raw viewport-size event type shared by live input targets."""

IMGUI_RAW_INPUT_CAPABILITIES = (
    UserInputCapability(
        event_type="key_down",
        input_modality="keyboard",
        payload_fields=frozenset({"key"}),
    ),
    UserInputCapability(
        event_type="key_up",
        input_modality="keyboard",
        payload_fields=frozenset({"key"}),
    ),
    UserInputCapability(
        event_type=POINTER_MOVE_EVENT_TYPE,
        input_modality="pointer",
        payload_fields=frozenset({"x", "y"}),
    ),
    UserInputCapability(
        event_type=POINTER_BUTTON_EVENT_TYPE,
        input_modality="pointer",
        payload_fields=frozenset({"button", "pressed"}),
    ),
    UserInputCapability(
        event_type=POINTER_WHEEL_EVENT_TYPE,
        input_modality="pointer",
        payload_fields=frozenset({"x", "y"}),
    ),
    UserInputCapability(
        event_type=TEXT_INPUT_EVENT_TYPE,
        input_modality="text",
        payload_fields=frozenset({"text"}),
    ),
    UserInputCapability(
        event_type=FOCUS_EVENT_TYPE,
        input_modality="window",
        payload_fields=frozenset({"focused"}),
    ),
    UserInputCapability(
        event_type=VIEWPORT_EVENT_TYPE,
        input_modality="window",
        payload_fields=frozenset({"width", "height"}),
    ),
)
"""Raw capabilities understood by :class:`ImGuiInputCanonicalizer`."""

IMGUI_RAW_INPUT_SCHEMA = UserInputSchema(
    capabilities=IMGUI_RAW_INPUT_CAPABILITIES,
    description="Transport-neutral keyboard, pointer, text, and viewport events.",
)
"""Complete raw input schema accepted by the server-side ImGui host."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ImGuiInputEvent:
    """One canonical event ready for an ImGui input adapter."""

    timestamp_s: float
    """Seconds since the presentation session started."""

    kind: ImGuiInputKind
    """ImGui-facing event category."""

    payload: Mapping[str, Any] = field(default_factory=dict)
    """Normalized event values consumed by :class:`ImGuiInputSink`."""

    source: str | None = None
    """Input source retained for diagnostics and replay."""

    source_event_id: str | None = None
    """Optional source identifier retained across canonicalization."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and >= 0.")
        object.__setattr__(self, "payload", freeze_mapping(self.payload))


@runtime_checkable
class ImGuiInputSink(Protocol):
    """Consume canonical input through Dear ImGui IO-style methods."""

    def add_mouse_pos_event(self, x: float, y: float) -> None: ...

    def add_mouse_button_event(self, button: int, pressed: bool) -> None: ...

    def add_mouse_wheel_event(self, x: float, y: float) -> None: ...

    def add_key_event(self, key: object, pressed: bool) -> None: ...

    def add_input_characters_utf8(self, text: str) -> None: ...

    def add_focus_event(self, focused: bool) -> None: ...

    def set_display_size(self, width: float, height: float) -> None: ...


class ImGuiInputCanonicalizer:
    """Normalize raw transport events into Dear ImGui input semantics."""

    def canonicalize(
        self,
        user_inputs: UserInputs,
        *,
        source_schema: UserInputSchema,
        window: TimeWindow | None = None,
        display_size: tuple[float, float] | None = None,
    ) -> tuple[ImGuiInputEvent, ...]:
        """Convert one raw input batch into ordered ImGui events.

        Args:
            user_inputs: Raw events supplied by one input target.
            source_schema: Capabilities declared by that input target.
            window: Optional half-open event-selection window.
            display_size: Current framebuffer size used to expand normalized
                pointer coordinates.

        Returns:
            Canonical events in source timestamp order.

        Raises:
            ValueError: An event is invalid or normalized coordinates lack a
                usable display size.
        """
        selected = user_inputs if window is None else user_inputs.window(window)
        canonical: list[ImGuiInputEvent] = []
        for event in selected.events:
            source_schema.validate_event(event)
            converted = self._canonicalize_event(event, display_size=display_size)
            if converted is not None:
                canonical.append(converted)
        return tuple(canonical)

    def _canonicalize_event(
        self,
        event: UserInputEvent,
        *,
        display_size: tuple[float, float] | None,
    ) -> ImGuiInputEvent | None:
        payload = event.payload
        canonical_payload: Mapping[str, Any]
        kind: ImGuiInputKind
        if event.event_type in {"key_down", "key_up"}:
            key = _normalize_imgui_key(str(payload["key"]))
            if not key:
                raise ValueError("Keyboard events require a non-empty key.")
            kind = "key"
            canonical_payload = {
                "key": key,
                "pressed": event.event_type == "key_down",
            }
        elif event.event_type == POINTER_MOVE_EVENT_TYPE:
            x = _finite_float(payload["x"], label="pointer x")
            y = _finite_float(payload["y"], label="pointer y")
            coordinate_space = str(payload.get("coordinate_space", "pixels"))
            if coordinate_space == "normalized":
                width, height = _require_display_size(display_size)
                x *= width
                y *= height
            elif coordinate_space != "pixels":
                raise ValueError(
                    "pointer coordinate_space must be 'pixels' or 'normalized'."
                )
            kind = "mouse_position"
            canonical_payload = {"x": x, "y": y}
        elif event.event_type == POINTER_BUTTON_EVENT_TYPE:
            kind = "mouse_button"
            canonical_payload = {
                "button": _nonnegative_int(payload["button"], label="mouse button"),
                "pressed": _bool(payload["pressed"], label="mouse pressed"),
            }
        elif event.event_type == POINTER_WHEEL_EVENT_TYPE:
            kind = "mouse_wheel"
            canonical_payload = {
                "x": _finite_float(payload["x"], label="wheel x"),
                "y": _finite_float(payload["y"], label="wheel y"),
            }
        elif event.event_type == TEXT_INPUT_EVENT_TYPE:
            text = str(payload["text"])
            if not text:
                return None
            kind = "text"
            canonical_payload = {"text": text}
        elif event.event_type == FOCUS_EVENT_TYPE:
            kind = "focus"
            canonical_payload = {
                "focused": _bool(payload["focused"], label="window focused")
            }
        elif event.event_type == VIEWPORT_EVENT_TYPE:
            width = _positive_float(payload["width"], label="viewport width")
            height = _positive_float(payload["height"], label="viewport height")
            kind = "display_size"
            canonical_payload = {"width": width, "height": height}
        else:
            return None
        return ImGuiInputEvent(
            timestamp_s=event.timestamp_s,
            kind=kind,
            payload=canonical_payload,
            source=event.source,
            source_event_id=event.source_event_id,
        )


class ImGuiInputRouter:
    """Feed canonical events into one Dear ImGui IO-compatible sink."""

    def __init__(
        self,
        sink: ImGuiInputSink,
        *,
        key_resolver: Callable[[str], object],
    ) -> None:
        self._sink = sink
        self._key_resolver = key_resolver

    def route(self, events: Sequence[ImGuiInputEvent]) -> None:
        """Apply ordered canonical events to the configured sink."""
        for event in events:
            payload = event.payload
            if event.kind == "mouse_position":
                self._sink.add_mouse_pos_event(float(payload["x"]), float(payload["y"]))
            elif event.kind == "mouse_button":
                self._sink.add_mouse_button_event(
                    int(payload["button"]), bool(payload["pressed"])
                )
            elif event.kind == "mouse_wheel":
                self._sink.add_mouse_wheel_event(
                    float(payload["x"]), float(payload["y"])
                )
            elif event.kind == "key":
                self._sink.add_key_event(
                    self._key_resolver(str(payload["key"])),
                    bool(payload["pressed"]),
                )
            elif event.kind == "text":
                self._sink.add_input_characters_utf8(str(payload["text"]))
            elif event.kind == "focus":
                self._sink.add_focus_event(bool(payload["focused"]))
            else:
                self._sink.set_display_size(
                    float(payload["width"]), float(payload["height"])
                )


class RawUIInputMailbox:
    """Fan raw input from transport threads into the presentation thread."""

    def __init__(self, *, source_schema: UserInputSchema) -> None:
        self.source_schema = source_schema
        self._lock = threading.Lock()
        self._pending: list[tuple[int, UserInputEvent]] = []
        self._next_sequence = 0

    def publish(self, event: UserInputEvent) -> None:
        """Validate and enqueue one raw event without blocking presentation."""
        self.source_schema.validate_event(event)
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._pending.append((sequence, event))

    def drain(self, *, through_s: float | None = None) -> UserInputs:
        """Remove and return pending events up to an optional timestamp."""
        if through_s is not None and (not math.isfinite(through_s) or through_s < 0):
            raise ValueError("through_s must be finite and >= 0 when set.")
        with self._lock:
            selected: list[tuple[int, UserInputEvent]] = []
            retained: list[tuple[int, UserInputEvent]] = []
            for queued in self._pending:
                if through_s is None or queued[1].timestamp_s <= through_s:
                    selected.append(queued)
                else:
                    retained.append(queued)
            self._pending = retained
        selected.sort(key=lambda queued: (queued[1].timestamp_s, queued[0]))
        return UserInputs(events=tuple(event for _, event in selected))

    def clear(self) -> None:
        """Discard every pending raw event."""
        with self._lock:
            self._pending.clear()


class ImGuiInputSession:
    """Own the per-client raw-input queue consumed by an ImGui UI thread.

    Input targets call :meth:`publish` from their event-loop threads. The
    presentation thread calls :meth:`pump` before each UI frame. Model input
    collection remains independent and may consume the same source event.
    """

    def __init__(
        self,
        sink: ImGuiInputSink,
        *,
        source_schema: UserInputSchema,
        key_resolver: Callable[[str], object],
    ) -> None:
        self.mailbox = RawUIInputMailbox(source_schema=source_schema)
        self._canonicalizer = ImGuiInputCanonicalizer()
        self._router = ImGuiInputRouter(sink, key_resolver=key_resolver)

    def publish(self, event: UserInputEvent) -> None:
        """Publish one target event without waiting for model generation."""
        self.mailbox.publish(event)

    def pump(
        self,
        *,
        display_size: tuple[float, float] | None = None,
        through_s: float | None = None,
    ) -> tuple[ImGuiInputEvent, ...]:
        """Canonicalize and route pending events on the UI thread."""
        inputs = self.mailbox.drain(through_s=through_s)
        events = self._canonicalizer.canonicalize(
            inputs,
            source_schema=self.mailbox.source_schema,
            display_size=display_size,
        )
        self._router.route(events)
        return events


def merged_user_input_schema(
    *schemas: UserInputSchema,
    description: str = "",
) -> UserInputSchema:
    """Merge source schemas while preserving capability order."""
    capabilities: list[UserInputCapability] = []
    seen: set[tuple[str, str | None, frozenset[str]]] = set()
    snapshot_fields = []
    seen_snapshot_fields: set[str] = set()
    event_types: set[str] = set()
    for schema in schemas:
        event_types.update(schema.event_types)
        for capability in schema.capabilities:
            key = (
                capability.event_type,
                capability.input_modality,
                capability.payload_fields,
            )
            if key not in seen:
                seen.add(key)
                capabilities.append(capability)
        for input_field in schema.snapshot_fields:
            if input_field.name not in seen_snapshot_fields:
                seen_snapshot_fields.add(input_field.name)
                snapshot_fields.append(input_field)
    return UserInputSchema(
        event_types=frozenset(event_types),
        snapshot_fields=tuple(snapshot_fields),
        capabilities=tuple(capabilities),
        description=description,
    )


def _normalize_imgui_key(key: str) -> str:
    if key == " ":
        return "space"
    normalized = key.strip().lower().replace("-", "_")
    aliases = {
        "alt": "left_alt",
        "arrowdown": "down_arrow",
        "arrowleft": "left_arrow",
        "arrowright": "right_arrow",
        "arrowup": "up_arrow",
        "control": "left_ctrl",
        "ctrl": "left_ctrl",
        "esc": "escape",
        "pagedown": "page_down",
        "pageup": "page_up",
        "return": "enter",
        "shift": "left_shift",
        "super": "left_super",
    }
    return aliases.get(normalized, normalized)


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite.")
    return normalized


def _positive_float(value: object, *, label: str) -> float:
    normalized = _finite_float(value, label=label)
    if normalized <= 0:
        raise ValueError(f"{label} must be > 0.")
    return normalized


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    if value < 0:
        raise ValueError(f"{label} must be >= 0.")
    return value


def _bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _require_display_size(
    display_size: tuple[float, float] | None,
) -> tuple[float, float]:
    if display_size is None:
        raise ValueError("Normalized pointer input requires display_size.")
    return (
        _positive_float(display_size[0], label="display width"),
        _positive_float(display_size[1], label="display height"),
    )


__all__ = [
    "FOCUS_EVENT_TYPE",
    "IMGUI_RAW_INPUT_CAPABILITIES",
    "IMGUI_RAW_INPUT_SCHEMA",
    "POINTER_BUTTON_EVENT_TYPE",
    "POINTER_MOVE_EVENT_TYPE",
    "POINTER_WHEEL_EVENT_TYPE",
    "TEXT_INPUT_EVENT_TYPE",
    "VIEWPORT_EVENT_TYPE",
    "ImGuiInputCanonicalizer",
    "ImGuiInputEvent",
    "ImGuiInputRouter",
    "ImGuiInputSink",
    "RawUIInputMailbox",
    "merged_user_input_schema",
    "ImGuiInputSession",
]
