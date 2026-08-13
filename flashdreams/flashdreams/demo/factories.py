# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Application output-sink factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flashdreams.demo.io import InputHandler, IOFactory, OutputSink, SessionInfo
from flashdreams.demo.local_input import SlangPyLocalInputHandler
from flashdreams.demo.local_window import LocalWindowInputBridge
from flashdreams.demo.outputs import (
    LocalWindowOutputSink,
    Mp4OutputSink,
    WebRTCOutputSink,
)
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime.inputs import CanonicalInputs, CanonicalInputSchema

if TYPE_CHECKING:
    from flashdreams.serving.webrtc.services import WebRTCOutputBridge


@dataclass(slots=True)
class NullInputHandler(InputHandler):
    """Provide an empty canonical input state."""

    opened: bool = False
    """Whether the handler is open for the current session."""

    def open(
        self,
        session_info: SessionInfo,
    ) -> None:
        """Open the empty handler for one application session."""
        del session_info
        self.opened = True

    def current_inputs(self) -> CanonicalInputs:
        """Return an empty canonical input state."""
        if not self.opened:
            raise RuntimeError("Cannot fetch inputs from a closed input handler.")
        return CanonicalInputs()

    def close(self) -> None:
        """Close the empty input handler."""
        self.opened = False


@dataclass(frozen=True, slots=True)
class CallableIOFactory(IOFactory):
    """Create handlers and sinks through caller-provided factories."""

    input_factory: Callable[[CanonicalInputSchema], InputHandler]
    """Create an input handler bound to an application schema."""

    output_factory: Callable[[], OutputSink]
    """Create an output sink for one application run."""

    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create an input handler for ``input_schema``."""
        return self.input_factory(input_schema)

    def create_output_sink(self) -> OutputSink:
        """Create an output sink for one application run."""
        return self.output_factory()


@dataclass(frozen=True, slots=True)
class ProvidedIOFactory(IOFactory):
    """Expose caller-owned input and output objects."""

    input_handler: InputHandler
    """Caller-owned canonical input handler."""

    output_sink: OutputSink
    """Caller-owned output sink."""

    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create an input handler for ``input_schema``."""
        del input_schema
        return self.input_handler

    def create_output_sink(self) -> OutputSink:
        """Create an output sink for one application run."""
        return self.output_sink


@dataclass(frozen=True, slots=True)
class LocalWindowIOFactory(IOFactory):
    """Create SlangPy input handlers and local-window output sinks."""

    title: str = "FlashDreams"
    """Local window title."""

    fps: float | None = None
    """Playback rate; ``None`` uses application session metadata."""

    presenter_factory: Callable[..., Any] | None = None
    """Optional native presenter factory for embedded hosts and tests."""

    _bridge: LocalWindowInputBridge = field(
        default_factory=LocalWindowInputBridge,
        init=False,
        repr=False,
        compare=False,
    )
    """Shared callback bridge for the input handler and window presenter."""

    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create an input handler for ``input_schema``."""
        handler = SlangPyLocalInputHandler(
            input_schema,
            process_events=self._bridge.process_events,
        )
        self._bridge.bind_handler(handler)
        return handler

    def create_output_sink(self) -> OutputSink:
        """Create an output sink for one application run."""
        return LocalWindowOutputSink(
            title=self.title,
            fps=self.fps,
            presenter_factory=self.presenter_factory,
            presenter_opened=self._bridge.bind_presenter,
        )


@dataclass(frozen=True, slots=True)
class Mp4IOFactory(IOFactory):
    """Create empty input handlers and MP4 artifact output sinks."""

    output_path: Path
    """Destination MP4 path."""

    fps: int | float | None = None
    """Output frame rate; ``None`` uses application session metadata."""

    output_layout: VideoTensorLayout | None = None
    """Required video layout; ``None`` uses application session metadata."""

    move_to_cpu: bool = True
    """Whether to move collected chunks to CPU memory immediately."""

    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create an input handler for ``input_schema``."""
        del input_schema
        return NullInputHandler()

    def create_output_sink(self) -> OutputSink:
        """Create an output sink for one application run."""
        return Mp4OutputSink(
            output_path=self.output_path,
            fps=self.fps,
            output_layout=self.output_layout,
            move_to_cpu=self.move_to_cpu,
        )


@dataclass(frozen=True, slots=True)
class WebRTCIOFactory(IOFactory):
    """Create application I/O objects owned by one WebRTC peer."""

    bridge_factory: Callable[[], WebRTCOutputBridge]
    """Create the transport bridge owned by a peer connection."""

    input_factory: Callable[[CanonicalInputSchema], InputHandler] = (
        lambda _schema: NullInputHandler()
    )
    """Create the peer input handler bound to an application schema."""

    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create an input handler for ``input_schema``."""
        return self.input_factory(input_schema)

    def create_output_sink(self) -> OutputSink:
        """Create an output sink for one application run."""
        return WebRTCOutputSink(bridge=self.bridge_factory())


__all__ = [
    "CallableIOFactory",
    "LocalWindowIOFactory",
    "Mp4IOFactory",
    "NullInputHandler",
    "ProvidedIOFactory",
    "WebRTCIOFactory",
]
