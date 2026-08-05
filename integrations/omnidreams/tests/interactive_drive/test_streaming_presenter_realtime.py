# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omnidreams.interactive_drive.browser_presenter import NativeHudBrowserPresenter
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.slangpy_hud_presenter import (
    HUD_PANEL_WIDTH,
    NVIDIA_GREEN,
)
from omnidreams.interactive_drive.streaming_presenter import (
    _as_rgb_host_uint8,
    _LatestFramePublisher,
    _publish_if_open,
    _wait_for_bus_frame,
)
from omnidreams.interactive_drive.types import PresentedFrame

pytestmark = pytest.mark.ci_cpu

from flashdreams.serving.realtime.frame_bus import LatestFrameBus


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


def test_latest_frame_publisher_drops_stale_pending_frames() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    latest_published = threading.Event()
    published: list[int] = []

    def publish(frame: object) -> None:
        value = int(frame)
        if value == 1:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        published.append(value)
        if value == 3:
            latest_published.set()

    publisher = _LatestFramePublisher(publish)
    try:
        publisher.submit(1)
        assert first_started.wait(timeout=1.0)
        publisher.submit(2)
        publisher.submit(3)
        release_first.set()
        assert latest_published.wait(timeout=1.0)
    finally:
        publisher.close()

    assert published == [1, 3]


def test_browser_presenter_preserves_native_hud_layout_when_model_starts() -> None:
    frames: list[np.ndarray] = []
    args = SimpleNamespace(
        scene=Path("/tmp/native-parity.usdz"),
        variant="default",
        bev=True,
        bev_resolution="1024x1024",
        bev_height_m=75.0,
        bev_fov_deg=60.0,
        bev_tilt_deg=0.0,
    )
    scene = SimpleNamespace(
        label="Native parity",
        path=args.scene,
        variants=("default",),
        variant_paths={},
        thumbnail=None,
    )
    assets = SimpleNamespace(
        steering_wheel=None,
        throttle_pressed=None,
        throttle_unpressed=None,
        brake_pressed=None,
        brake_unpressed=None,
    )
    presenter = NativeHudBrowserPresenter(
        RasterConfig(width=1280, height=704),
        KeyboardState(),
        args=args,
        scene_options=(scene,),
        control_assets=assets,
        frame_sink=frames.append,
    )
    try:
        rgb = np.zeros((704, 1280, 3), dtype=np.uint8)
        loading_frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=rgb,
            depth_host_f32=None,
            status_message="Loading scene...",
        )
        presenter.set_engine_active(True)
        presenter.present_frame(loading_frame, "model_rgb")
        loading_shape = frames[-1].shape

        bev = np.zeros((1024, 1024, 3), dtype=np.uint8)
        model_frame = PresentedFrame(
            timestamp_us=1,
            rgb_host_uint8=rgb,
            depth_host_f32=None,
            model_rgb_host_uint8=rgb,
            bev_host_uint8=bev,
        )
        presenter.present_frame(model_frame, "model_rgb")
        rendered = frames[-1]
        assert loading_shape == (1080, 1920, 3)
        assert rendered.shape == loading_shape
        assert presenter._latest_bev_source is bev
        assert presenter._bev_panel_target_size[1] > 0
        np.testing.assert_array_equal(
            rendered[100, rendered.shape[1] - HUD_PANEL_WIDTH],
            np.asarray(NVIDIA_GREEN, dtype=np.uint8),
        )
    finally:
        presenter.close()
