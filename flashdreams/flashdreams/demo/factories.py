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

"""Application I/O factories and the default empty input sink."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flashdreams.demo.io import InputSink, IOFactory, OutputSink, SessionInfo
from flashdreams.demo.outputs import (
    LocalWindowOutputSink,
    Mp4OutputSink,
)
from flashdreams.infra.postprocess import VideoTensorLayout


@dataclass(slots=True)
class NullInputSink(InputSink):
    """Supply no dynamic input to an application session."""

    opened: bool = False
    """Whether the input sink is open."""

    def open(self, session_info: SessionInfo) -> None:
        """Open the empty input source for a session."""
        del session_info
        self.opened = True

    def read(self) -> None:
        """Return ``None`` because no input is available."""
        if not self.opened:
            raise RuntimeError("Cannot read from a closed input sink.")
        return None

    def close(self) -> None:
        """Close the empty input source."""
        self.opened = False


@dataclass(frozen=True, slots=True)
class CallableIOFactory(IOFactory):
    """Create sinks through caller-provided zero-argument factories."""

    input_factory: Callable[[], InputSink]
    """Factory for isolated input sinks."""

    output_factory: Callable[[], OutputSink]
    """Factory for isolated output sinks."""

    def create_input_sink(self) -> InputSink:
        """Create an input sink."""
        return self.input_factory()

    def create_output_sink(self) -> OutputSink:
        """Create an output sink."""
        return self.output_factory()


@dataclass(frozen=True, slots=True)
class ProvidedIOFactory(IOFactory):
    """Expose caller-owned sink instances through the factory boundary."""

    input_sink: InputSink
    """Caller-owned input sink."""

    output_sink: OutputSink
    """Caller-owned output sink."""

    def create_input_sink(self) -> InputSink:
        """Return the caller-owned input sink."""
        return self.input_sink

    def create_output_sink(self) -> OutputSink:
        """Return the caller-owned output sink."""
        return self.output_sink


@dataclass(frozen=True, slots=True)
class LocalWindowIOFactory(IOFactory):
    """Create empty input and local-window output sinks."""

    title: str = "FlashDreams"
    """Local window title."""

    fps: float | None = None
    """Playback rate; ``None`` uses application session metadata."""

    presenter_factory: Callable[..., Any] | None = None
    """Optional native presenter factory for embedded hosts and tests."""

    def create_input_sink(self) -> InputSink:
        """Create an empty input sink."""
        return NullInputSink()

    def create_output_sink(self) -> OutputSink:
        """Create a local-window output sink."""
        return LocalWindowOutputSink(
            title=self.title,
            fps=self.fps,
            presenter_factory=self.presenter_factory,
        )


@dataclass(frozen=True, slots=True)
class Mp4IOFactory(IOFactory):
    """Create empty input and MP4 artifact output sinks."""

    output_path: Path
    """Destination MP4 path."""

    fps: int | float | None = None
    """Output frame rate; ``None`` uses application session metadata."""

    output_layout: VideoTensorLayout | None = None
    """Required video layout; ``None`` uses application session metadata."""

    move_to_cpu: bool = True
    """Whether to move collected chunks to CPU memory immediately."""

    def create_input_sink(self) -> InputSink:
        """Create an empty input sink."""
        return NullInputSink()

    def create_output_sink(self) -> OutputSink:
        """Create an MP4 output sink."""
        return Mp4OutputSink(
            output_path=self.output_path,
            fps=self.fps,
            output_layout=self.output_layout,
            move_to_cpu=self.move_to_cpu,
        )


__all__ = [
    "CallableIOFactory",
    "LocalWindowIOFactory",
    "Mp4IOFactory",
    "NullInputSink",
    "ProvidedIOFactory",
]
