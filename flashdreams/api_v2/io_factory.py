# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application I/O factory protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.core_v2.session_desc import SessionDesc

from .input_handler import InputHandler
from .output_sink import OutputSink


@runtime_checkable
class IOFactory(Protocol):
    """Create isolated application input handling and output delivery."""

    def __init__(self, session_desc: SessionDesc) -> None:
        """Initialize the factory according to session information."""
        ...

    @abstractmethod
    def create_input_handler(self) -> InputHandler:
        """Create a handler for the application-declared canonical inputs."""
        ...

    @abstractmethod
    def create_output_sink(self) -> OutputSink:
        """Create the output sink for one application run."""
        ...
