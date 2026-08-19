# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC client window for the v2 runtime."""

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class WebRTCClientWindow(IClientWindow):
    """Adapt a standalone WebRTC server to the v2 client-window interface."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        """
        Args:
            host: Interface on which the HTTP server listens.
            port: Listening port. Zero asks the operating system to choose one.
            startup_timeout_seconds: Maximum time to wait for server startup.
        """
        self.server = WebRTCServer(
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
        )

    def open(self, session_desc: SessionDesc) -> None:
        """Configure WebRTC output for the session.

        Args:
            session_desc: Resolved dimensions, frame rate, and tensor layout.
        """
        self.server.open(session_desc)

    def get_user_input_events(self) -> UserInputEvents:
        """Drain and return buffered browser events in timestamp order."""
        return self.server.get_user_input_events()

    def write(self, result: StepResult) -> None:
        """Deliver one generated result to the browser.

        Args:
            result: Generated frames matching the opened session.
        """
        self.server.write(result)

    def close(self) -> None:
        """Close the WebRTC connection and server."""
        self.server.close()
