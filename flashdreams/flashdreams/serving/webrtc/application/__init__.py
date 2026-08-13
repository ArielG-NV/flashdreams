# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC hosting adapter for FlashDreams applications."""

from flashdreams.serving.webrtc.application.factory import WebRTCIOFactory
from flashdreams.serving.webrtc.application.server import serve_application_webrtc
from flashdreams.serving.webrtc.application.session_manager import (
    ApplicationWebRTCSessionManager,
    BufferedTrackOutputBridge,
)

__all__ = [
    "ApplicationWebRTCSessionManager",
    "BufferedTrackOutputBridge",
    "WebRTCIOFactory",
    "serve_application_webrtc",
]
