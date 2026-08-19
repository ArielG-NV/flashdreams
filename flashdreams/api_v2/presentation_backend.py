# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation backend protocol."""

from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod

from .input_source import InputSource
from .output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc

class IPresentationBackend(InputSource, OutputSink, ABC):
    """Handle application input and output for one presentation backend."""
    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Subclasses must provide this variable or property."""
        ...