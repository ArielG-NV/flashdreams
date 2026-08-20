# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 client-window factory."""

import argparse

import pytest

pytestmark = pytest.mark.ci_cpu

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")

from flashdreams.runtime_v2.client_window_factory import create_client_window
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


def test_create_client_window_selects_webrtc() -> None:
    window = create_client_window(
        argparse.Namespace(mode="webrtc", host="127.0.0.1", port=0)
    )
    try:
        assert isinstance(window, WebRTCClientWindow)
    finally:
        window.close()


def test_create_client_window_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_client_window(argparse.Namespace(mode="local"))
