# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC, abstractmethod

from flashdreams.runtime_v2.session_desc import SessionDesc

from .input_source import InputSource
from .output_sink import OutputSink


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle input and output for one client window.

    Use is generally: calls `open` at application start (when outputs are ready to be written),
    reads input, generates, then writes results. Finally we call `close` when outputs are no longer
    accepted (generally, application closed or is resetting).
    """

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the session description for this window."""
        ...
