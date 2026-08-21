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

"""ImGui demo for messages sent from model generation to the UI worker."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import PresentationMode, StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .runner import run_demo

_IMGUI_THREAD_ID = 1
"""Auxiliary worker identifier for the UI layer."""

_W_NOT_PRESSED = "W is not Pressed"
"""UI status shown before model generation receives a W key-down event."""

_W_PRESSED = "W is Pressed"
"""UI status sent after model generation receives a W key-down event."""


@dataclass(slots=True)
class MessageState:
    """Status text owned exclusively by the UI worker."""

    status: str = _W_NOT_PRESSED
    """Text displayed beneath the keyboard prompt."""


class MessageImGUIThread(ImGUIThread[MessageState]):
    """Display state updated through model-generation messages."""

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> None:
        """Prompt for W input and draw the current message-owned state."""
        del step_index, events
        imgui.set_next_window_pos((16, 16), imgui.Cond_.once)
        imgui.set_next_window_size((360, 110), imgui.Cond_.once)
        imgui.begin("Model-to-UI message demo")
        imgui.text("Press W to send a message from model generation.")
        imgui.text(self.state.status)
        imgui.end()

    def reset(self) -> None:
        """Restore the initial status for a new generation."""
        self.state.status = _W_NOT_PRESSED
        super().reset()


class MessageSession(ISession):
    """Send a UI-state operation when model generation receives W input."""

    def __init__(self, session_desc: SessionDesc) -> None:
        """Configure one message demo session.

        Args:
            session_desc: Output dimensions and worker frequencies.

        Raises:
            ValueError: ``session_desc`` does not request ``tchw`` output.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The message demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._output = torch.empty(
            (1, 3, session_desc.video_height, session_desc.video_width),
            dtype=torch.float32,
        )
        self._ui_thread = MessageImGUIThread(
            state=MessageState(),
            frequency=session_desc.frames_per_second_for_ui,
            output_layout=session_desc.output_layout,
            width=session_desc.video_width,
            height=session_desc.video_height,
        )

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the message-receiving UI worker."""
        self.register_thread(self._ui_thread, _IMGUI_THREAD_ID)

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Send a UI update when this iteration receives a W state change."""
        state = _last_w_key_state(events)
        if state is KeyboardInputState.Pressed:
            self._ui_thread.invoke_async(_IMGUI_THREAD_ID, _mark_w_pressed)
        elif state is KeyboardInputState.Released:
            self._ui_thread.invoke_async(_IMGUI_THREAD_ID, _mark_w_released)
        return StepResult(
            step_index=step_index,
            output=self._output,
            frame_count=1,
            output_layout=self._session_desc.output_layout,
            presentation_mode=PresentationMode.disablePresentation,
        )

    def reset(self) -> None:
        """Leave UI-state reset to its owning worker."""
        return


class MessageApplication(IApplication):
    """Create sessions that demonstrate model-to-UI messages."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The message demo takes no application arguments.")

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized message demo session."""
        return MessageSession(session_desc)


def _last_w_key_state(events: UserInputEvents) -> KeyboardInputState | None:
    """Return the latest W transition in ``events``, if one exists."""
    state = None
    for event in events.get_events():
        data = event.get_event_data()
        if isinstance(data, KeyboardUserInputEventData) and data.key.lower() == "w":
            state = data.state
    return state


def _mark_w_pressed(state: MessageState) -> None:
    state.status = _W_PRESSED


def _mark_w_released(state: MessageState) -> None:
    state.status = _W_NOT_PRESSED


def create_app() -> IApplication:
    """Return a new model-to-UI message application."""
    return MessageApplication()


def main(commandline_args: Sequence[str] | None = None) -> int:
    """Serve the message demo until the browser disconnects."""
    return run_demo(
        create_app,
        commandline_args,
        program="imgui-message-webrtc",
        description="Show model-generation messages updating ImGui state.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
