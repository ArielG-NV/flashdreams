# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation backend protocol."""

from typing import Protocol, runtime_checkable

from .input_handler import InputHandler
from .output_sink import OutputSink


@runtime_checkable
class IPresentationSurface(InputHandler, OutputSink, Protocol):
    """Handle application input and output for one presentation backend."""
