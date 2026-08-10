# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from typing import cast

import numpy as np
from omnidreams.interactive_drive.streaming_presenter import (
    MJPEGStreamingPresenter,
    _INDEX_HTML,
    _as_rgb_host_uint8,
    _make_handler,
    _publish_if_open,
    _wait_for_bus_frame,
)

from flashdreams.serving.realtime.frame_bus import LatestFrameBus
from flashdreams.serving.realtime.input import WebControllerState


def test_streaming_presenter_materializes_lazy_rgba_frames() -> None:
    class LazyFrame:
        def to_numpy(self) -> np.ndarray:
            return np.array(
                [[[1, 2, 3, 255], [4, 5, 6, 255]]],
                dtype=np.uint8,
            )

    frame = _as_rgb_host_uint8(LazyFrame())

    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(
        frame,
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
    )


def test_streaming_presenter_publishes_jpeg_on_latest_frame_bus() -> None:
    bus = LatestFrameBus[bytes]()

    _publish_if_open(bus, b"jpeg", stop_event=threading.Event())

    latest = bus.latest()
    assert latest is not None
    assert latest.payload == b"jpeg"
    assert latest.count == 1


def test_streaming_presenter_frame_wait_returns_none_after_bus_close() -> None:
    bus = LatestFrameBus[bytes]()
    bus.publish(b"old")
    bus.close()

    frame = _wait_for_bus_frame(
        bus,
        last_seen_count=1,
        stop_event=threading.Event(),
    )

    assert frame is None


def test_mjpeg_page_posts_browser_controller_state() -> None:
    assert "navigator.getGamepads" in _INDEX_HTML
    assert "fetch('/controller'" in _INDEX_HTML
    assert "Controller / WASD / Arrows = Drive" in _INDEX_HTML


def test_mjpeg_controller_endpoint_forwards_json_payload() -> None:
    class Presenter:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def _apply_controller(self, payload: dict[str, object]) -> None:
            self.payloads.append(payload)

    presenter = Presenter()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _make_handler(cast(MJPEGStreamingPresenter, presenter)),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", int(server.server_address[1]), timeout=2.0
    )
    payload = {"steering": 0.25, "throttle": 0.8, "brake": 0.0}
    try:
        connection.request(
            "POST",
            "/controller",
            body=json.dumps(payload),
            headers={"content-type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)

    assert response.status == 204
    assert presenter.payloads == [payload]


def test_mjpeg_controller_overrides_keyboard_only_while_active_and_fresh() -> None:
    class Sink:
        def __init__(self) -> None:
            self.commands: list[tuple[float, float, float]] = []

        def set_drive(self, *, steer: float, throttle: float, brake: float) -> None:
            self.commands.append((steer, throttle, brake))

    class KeyboardDrive:
        def __init__(self) -> None:
            self.updates = 0

        def update(self) -> None:
            self.updates += 1

    presenter = object.__new__(MJPEGStreamingPresenter)
    presenter._controller_lock = threading.Lock()
    presenter._web_controller_state = WebControllerState(steering=0.5, throttle=0.75)
    presenter._web_controller_updated_s = time.monotonic()
    presenter._controller_sink = Sink()
    presenter._keyboard_drive = KeyboardDrive()

    presenter.process_events()

    assert presenter._controller_sink.commands == [(0.5, 0.75, 0.0)]
    assert presenter._keyboard_drive.updates == 0

    presenter._web_controller_updated_s = time.monotonic() - 2.0
    presenter.process_events()

    assert presenter._keyboard_drive.updates == 1
