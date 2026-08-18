# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for transport-neutral Dear ImGui input routing."""

from __future__ import annotations

from typing import Any

import pytest

from flashdreams.runtime import (
    IMGUI_RAW_INPUT_SCHEMA,
    ImGuiInputCanonicalizer,
    ImGuiInputRouter,
    ImGuiInputSession,
    RawUIInputMailbox,
    UserInputEvent,
    UserInputs,
)

pytestmark = pytest.mark.ci_cpu


class _Sink:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def add_mouse_pos_event(self, x: float, y: float) -> None:
        self.calls.append(("position", x, y))

    def add_mouse_button_event(self, button: int, pressed: bool) -> None:
        self.calls.append(("button", button, pressed))

    def add_mouse_wheel_event(self, x: float, y: float) -> None:
        self.calls.append(("wheel", x, y))

    def add_key_event(self, key: object, pressed: bool) -> None:
        self.calls.append(("key", key, pressed))

    def add_input_characters_utf8(self, text: str) -> None:
        self.calls.append(("text", text))

    def add_focus_event(self, focused: bool) -> None:
        self.calls.append(("focus", focused))

    def set_display_size(self, width: float, height: float) -> None:
        self.calls.append(("display", width, height))


def _event(timestamp_s: float, event_type: str, **payload: object) -> UserInputEvent:
    return UserInputEvent(
        timestamp_s=timestamp_s,
        event_type=event_type,
        payload=payload,
        source="test",
    )


def test_canonicalizer_normalizes_target_events_for_imgui() -> None:
    inputs = UserInputs(
        events=(
            _event(0.1, "pointer_move", x=0.25, y=0.5, coordinate_space="normalized"),
            _event(0.2, "pointer_button", button=1, pressed=True),
            _event(0.3, "pointer_wheel", x=-1.0, y=2.0),
            _event(0.4, "key_down", key="ArrowLeft"),
            _event(0.5, "text_input", text="hello"),
            _event(0.6, "focus", focused=True),
            _event(0.7, "viewport", width=1920, height=1080),
        )
    )

    events = ImGuiInputCanonicalizer().canonicalize(
        inputs,
        source_schema=IMGUI_RAW_INPUT_SCHEMA,
        display_size=(1920, 1080),
    )

    assert [event.kind for event in events] == [
        "mouse_position",
        "mouse_button",
        "mouse_wheel",
        "key",
        "text",
        "focus",
        "display_size",
    ]
    assert events[0].payload == {"x": 480.0, "y": 540.0}
    assert events[3].payload == {"key": "left_arrow", "pressed": True}


def test_imgui_input_session_routes_mailbox_on_ui_thread() -> None:
    sink = _Sink()
    session = ImGuiInputSession(
        sink,
        source_schema=IMGUI_RAW_INPUT_SCHEMA,
        key_resolver=lambda key: f"imgui:{key}",
    )
    session.publish(_event(0.1, "key_down", key="w"))
    session.publish(_event(0.2, "text_input", text="W"))

    routed = session.pump(display_size=(640, 480))

    assert len(routed) == 2
    assert sink.calls == [("key", "imgui:w", True), ("text", "W")]
    assert session.pump(display_size=(640, 480)) == ()


def test_raw_ui_mailbox_orders_events_and_drains_through_timestamp() -> None:
    mailbox = RawUIInputMailbox(source_schema=IMGUI_RAW_INPUT_SCHEMA)
    mailbox.publish(_event(0.3, "key_up", key="a"))
    mailbox.publish(_event(0.1, "key_down", key="a"))
    mailbox.publish(_event(0.3, "text_input", text="a"))

    first = mailbox.drain(through_s=0.2)
    remaining = mailbox.drain()

    assert [event.event_type for event in first.events] == ["key_down"]
    assert [event.event_type for event in remaining.events] == [
        "key_up",
        "text_input",
    ]


def test_router_calls_io_style_sink_methods() -> None:
    sink = _Sink()
    events = ImGuiInputCanonicalizer().canonicalize(
        UserInputs(
            events=(
                _event(0.1, "pointer_move", x=10, y=20),
                _event(0.2, "pointer_button", button=0, pressed=False),
                _event(0.3, "pointer_wheel", x=0, y=-1),
            )
        ),
        source_schema=IMGUI_RAW_INPUT_SCHEMA,
    )

    ImGuiInputRouter(sink, key_resolver=lambda key: key).route(events)

    assert sink.calls == [
        ("position", 10.0, 20.0),
        ("button", 0, False),
        ("wheel", 0.0, -1.0),
    ]
