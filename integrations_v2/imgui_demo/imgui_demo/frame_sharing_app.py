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

"""ImGui demo that displays the model thread's latest presented frame."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.presentation_cordinator import (
    PresentedFrame,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import PresentationMode, StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .app import DEFAULT_SESSION_DESC

_STEPS_PER_COLOR = 10
"""Model iterations that present each color before rotating."""

_RGB_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
"""Eight-bit RGB colors emitted by the model-generation-thread."""


@dataclass(frozen=True, slots=True)
class FrameSharingState:
    """Display geometry owned by the frame-sharing ImGui-thread."""

    image_size: tuple[float, float]
    """Width and height used to draw the model frame inside the UI window."""

    model_generation_thread_frame: PresentedFrame
    """Read-only handle to the model-generation-thread's presented frame."""


@dataclass(frozen=True, slots=True)
class FrameSharingModelState:
    """Output state owned by the model-generation-thread."""

    frames: tuple[torch.Tensor, ...]
    output_layout: VideoTensorLayout


class FrameSharingImGUIThread(ImGUIThread[FrameSharingState]):
    """Display the model-generation-thread's latest presented frame."""

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> None:
        """Draw the newest model frame inside an ImGui window."""
        del step_index, events
        image_width, image_height = self.state.image_size
        imgui.set_next_window_pos((16, 16), imgui.Cond_.once)
        imgui.set_next_window_size(
            (image_width + 32, image_height + 64),
            imgui.Cond_.once,
        )
        imgui.begin("Latest model frame")
        frame = self.state.model_generation_thread_frame.get()
        if frame is None:
            imgui.text("Waiting for the first presented model frame...")
        else:
            self.draw_frame(imgui, frame, self.state.image_size)
        imgui.end()


class FrameSharingModelThread(IThread[FrameSharingModelState]):
    """Publish rotating model frames from the model-generation-thread."""

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Return the color selected for this model iteration."""
        del events
        color_index = (step_index // _STEPS_PER_COLOR) % len(self.state.frames)
        return StepResult(
            step_index=step_index,
            output=self.state.frames[color_index],
            frame_count=1,
            output_layout=self.state.output_layout,
            presentation_mode=PresentationMode.HIDE_PRESENTATION,
        )

    def reset(self) -> None:
        return


class FrameSharingSession(ISession):
    """Rotate model colors while an ImGui-thread displays the latest frame."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one frame-sharing session.

        Args:
            session_desc: Output dimensions and user-visible-thread frequencies.
            device: Device used for model frames. The live UI requires ``"cuda"``;
                tests may inject ``"cpu"``.

        Raises:
            ValueError: ``session_desc`` does not request ``tchw`` output.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The frame-sharing demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._device = device
        self._session_desc = session_desc

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the model-generation-thread and frame-sharing ImGui-thread."""
        main_generation_thread_id = self.register_main_generation_thread(
            FrameSharingModelThread,
            state=FrameSharingModelState(
                frames=tuple(
                    _solid_frame(
                        rgb,
                        self._session_desc,
                        device=self._device,
                    )
                    for rgb in _RGB_COLORS
                ),
                output_layout=self._session_desc.output_layout,
            ),
        )
        imgui_thread_id = self.register_thread(
            FrameSharingImGUIThread,
            state=FrameSharingState(
                image_size=_image_size(self._session_desc),
                model_generation_thread_frame=self.get_presentation_cordinator().get_last_presented_frame(
                    main_generation_thread_id
                ),
            ),
            frequency=self._session_desc.frames_per_second_for_ui,
            output_layout=self._session_desc.output_layout,
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.set_layer_order_via_thread_id([main_generation_thread_id, imgui_thread_id])


class FrameSharingApplication(IApplication):
    """Create sessions that share model frames with an ImGui-thread."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The frame-sharing demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and thread frequencies."""
        return DEFAULT_SESSION_DESC

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized frame-sharing session."""
        return FrameSharingSession(session_desc)


def _solid_frame(
    rgb: tuple[int, int, int],
    session_desc: SessionDesc,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Create one normalized ``[1, C, H, W]`` solid-color frame."""
    color = torch.tensor(rgb, dtype=torch.float32, device=device).div(127.5).sub(1.0)
    return color.view(1, 3, 1, 1).expand(
        1,
        3,
        session_desc.video_height,
        session_desc.video_width,
    )


def _image_size(session_desc: SessionDesc) -> tuple[float, float]:
    """Fit the model frame inside the output while preserving its aspect ratio."""
    width = session_desc.video_width
    height = session_desc.video_height
    scale = min(max(1, width - 64) / width, max(1, height - 72) / height)
    return width * scale, height * scale


def create_app() -> IApplication:
    """Return a new frame-sharing application."""
    return FrameSharingApplication()
