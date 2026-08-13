# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server bootstrap for WebRTC-hosted FlashDreams applications."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files

from aiohttp import web

from flashdreams.serving.webrtc.application.session_manager import (
    ApplicationWebRTCSessionManager,
)
from flashdreams.serving.webrtc.server import create_packaged_webrtc_app


def serve_application_webrtc(
    application_slug: str,
    commandline_args: Sequence[str],
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run an installed application behind the packaged WebRTC browser UI.

    Args:
        application_slug: Installed application slug.
        commandline_args: Arguments forwarded to the application.
        host: Interface on which the HTTP server listens.
        port: TCP port on which the HTTP server listens.
    """
    manager = ApplicationWebRTCSessionManager(
        application_slug=application_slug,
        commandline_args=commandline_args,
    )
    request_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    app = create_packaged_webrtc_app(
        web_resource=files("flashdreams.serving.webrtc").joinpath("web"),
        session_manager=manager,
        request_session_url=f"http://{request_host}:{port}/request_session",
        preload_name=application_slug,
    )
    web.run_app(app, host=host, port=port)


__all__ = ["serve_application_webrtc"]
