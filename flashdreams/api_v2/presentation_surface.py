# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation backend protocol."""

from typing import Protocol, runtime_checkable

from .input_handler import InputHandler
from .output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc


@runtime_checkable
class IPresentationSurface(InputHandler, OutputSink, Protocol):
    """Handle application input and output for one presentation backend."""
    session_desc : SessionDesc
    def __init__(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc