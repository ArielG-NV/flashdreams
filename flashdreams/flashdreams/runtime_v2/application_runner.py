# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle runner for the v2 runtime."""

from collections.abc import Sequence

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session


class ApplicationRunner:
    """Create and run one application session against one client window."""

    def __init__(self, application: IApplication, client_window: IClientWindow) -> None:
        """
        Args:
            application: Long-lived application that creates the session.
            client_window: Window that supplies input and presents generated output.
        """
        self._application = application
        self._client_window = client_window

    def run(
        self,
        session_desc: SessionDesc,
        commandline_args: Sequence[str] = (),
    ) -> None:
        """Initialize the application, create one session, and run it.

        The application is closed before this method returns or raises.

        Args:
            session_desc: Output shape and timing requested for the session.
            commandline_args: Arguments owned and parsed by the application.
        """
        try:
            self._application.init(commandline_args)
            session = self._application.create_session(session_desc)
            run_session(session, self._client_window)
        finally:
            self._application.close()
