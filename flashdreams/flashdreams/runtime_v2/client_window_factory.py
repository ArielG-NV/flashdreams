# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create v2 client windows from runtime arguments."""

import argparse

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


def create_client_window(parsed_args: argparse.Namespace) -> IClientWindow:
    """Create the client window selected by the presentation mode.

    Args:
        parsed_args: Runtime arguments. Mode-specific fields are read only by
            the selected mode.

    Returns:
        Client window for the selected mode.

    Raises:
        ValueError: ``mode`` is unsupported.
    """
    if parsed_args.mode == "webrtc":
        return WebRTCClientWindow(host=parsed_args.host, port=parsed_args.port)
    raise ValueError(f"Unsupported client-window mode: {parsed_args.mode!r}.")
