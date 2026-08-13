# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC I/O factory for hosted FlashDreams applications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from flashdreams.demo.factories import NullInputSink
from flashdreams.demo.io import InputSink, IOFactory, OutputSink
from flashdreams.serving.webrtc.services import (
    WebRTCOutputBridge,
    WebRTCOutputSink,
)


@dataclass(frozen=True, slots=True)
class WebRTCIOFactory(IOFactory):
    """Create application sinks owned by one WebRTC peer."""

    bridge_factory: Callable[[], WebRTCOutputBridge]
    """Create the transport bridge owned by a peer connection."""

    input_factory: Callable[[], InputSink] = NullInputSink
    """Create the application input sink."""

    def create_input_sink(self) -> InputSink:
        """Create an input sink."""
        return self.input_factory()

    def create_output_sink(self) -> OutputSink:
        """Create an output sink connected to the peer bridge."""
        return WebRTCOutputSink(bridge=self.bridge_factory())


__all__ = ["WebRTCIOFactory"]
