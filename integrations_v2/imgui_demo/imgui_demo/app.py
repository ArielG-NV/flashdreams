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

"""Dear ImGui-only application for the v2 threaded runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.client_window_factory import create_client_window
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

_IMGUI_THREAD_ID = 1
"""Auxiliary worker identifier for the UI layer."""


@dataclass(slots=True)
class ImGUIDemoState:
    """Button state owned by the UI worker."""

    clicks: int = 0
    """Number of times the button has been activated."""
    text: str = ""
    """Text input."""


class DemoImGUIThread(ImGUIThread[ImGUIDemoState]):
    """Draw the demo widgets on an independent UI worker."""

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> None:
        """Draw one button and its click count."""
        del step_index, events
        imgui.set_next_window_pos((16, 16), imgui.Cond_.once)
        imgui.set_next_window_size((180, 80), imgui.Cond_.once)
        imgui.begin("ImGui demo")
        if imgui.button("Click"):
            self.state.clicks += 1

        imgui.same_line()
        imgui.text(f"{self.state.clicks}")
        changed, text = imgui.input_text("Text input", self.state.text)
        if changed:
            self.state.text = text
        imgui.end()

    def reset(self) -> None:
        """Restore widget values for a new generation."""
        self.state.clicks = 0
        super().reset()


class ImGUIDemoSession(ISession):
    """Run a disabled generation layer beneath one live ImGui layer."""

    def __init__(self, session_desc: SessionDesc) -> None:
        """Configure a demo session.

        Args:
            session_desc: Output dimensions and UI frequency.

        Raises:
            ValueError: ``session_desc`` does not request ``tchw`` output.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The ImGui demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._disabled_output = torch.empty(
            (1, 3, session_desc.video_height, session_desc.video_width),
            dtype=torch.float32,
        )

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the ImGui worker before the runtime freezes registration."""
        ui_thread = DemoImGUIThread(
            state=ImGUIDemoState(),
            frequency=self.session_desc.frames_per_second_for_ui,
            output_layout=self._session_desc.output_layout,
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_thread(ui_thread, _IMGUI_THREAD_ID)

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Return an explicitly disabled main-generation result."""
        del events
        return StepResult(
            step_index=step_index,
            output=self._disabled_output,
            frame_count=1,
            output_layout=self._session_desc.output_layout,
            disabled=True,
        )

    def reset(self) -> None:
        """Leave generation disabled while auxiliary workers reset themselves."""
        return


class ImGUIDemoApplication(IApplication):
    """Create ImGui-only sessions for the v2 runtime."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The ImGui demo takes no application arguments.")

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized ImGui demo session."""
        return ImGUIDemoSession(session_desc)


def create_app() -> IApplication:
    """Return a new ImGui demo application."""
    return ImGUIDemoApplication()


def _parse_args(commandline_args: Sequence[str] | None) -> argparse.Namespace:
    """Parse WebRTC runtime arguments."""
    parser = argparse.ArgumentParser(
        prog="imgui-demo-webrtc",
        description="Serve the v2 Dear ImGui input demo.",
    )
    parser.add_argument("--mode", choices=("webrtc",), default="webrtc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args(commandline_args)


def main(commandline_args: Sequence[str] | None = None) -> int:
    """Serve the ImGui demo until the browser disconnects."""
    args = _parse_args(commandline_args)
    window = create_client_window(args)
    app = create_app()
    if isinstance(window, WebRTCClientWindow):
        print(f"Open {window.server.url} in a browser.", flush=True)
    try:
        ApplicationRunner(app, window).run(
            SessionDesc(
                output_layout=VideoTensorLayout.tchw,
                frames_per_second_for_ui=args.fps,
                frames_per_second_for_step=args.fps,
                video_width=args.width,
                video_height=args.height,
            )
        )
    except KeyboardInterrupt:
        return 130
    finally:
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
