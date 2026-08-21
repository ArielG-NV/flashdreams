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

"""CPU tests for the v2 Dear ImGui demo integration."""

import pytest
import torch
from imgui_demo.app import (
    DemoImGUIThread,
    DemoModelThread,
    ImGUIDemoApplication,
    ImGUIDemoSession,
    ImGUIDemoState,
)
from imgui_demo.frame_sharing_app import (
    FrameSharingApplication,
    FrameSharingModelThread,
    FrameSharingSession,
)
from imgui_demo.message_app import (
    MessageApplication,
    MessageImGUIThread,
    MessageModelThread,
    MessageSession,
)
from numpy import uint64

from flashdreams.runtime_v2.application_registry import registered_application_slugs
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import PresentationMode
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _session() -> ImGUIDemoSession:
    return ImGUIDemoSession(
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            video_width=32,
            video_height=24,
        )
    )


def test_main_generation_disables_presentation() -> None:
    session = _session()
    session.init()
    model_generation_thread = session._ensure_thread_manager()._get_thread(0)
    assert isinstance(model_generation_thread, DemoModelThread)

    result = model_generation_thread.step(3, UserInputEvents([]))

    assert result.step_index == 3
    assert result.presentation_mode is PresentationMode.DISABLE_PRESENTATION
    assert result.output.shape == (1, 3, 24, 32)


@pytest.mark.parametrize(
    "application_type",
    (ImGUIDemoApplication, FrameSharingApplication, MessageApplication),
)
def test_applications_preserve_the_demo_session_defaults(
    application_type: type,
) -> None:
    session_desc = application_type().session_desc()

    assert session_desc == SessionDesc(video_width=640, video_height=480)


def test_all_imgui_applications_are_registered() -> None:
    assert {
        "imgui-demo",
        "imgui-frame-sharing",
        "imgui-message",
    }.issubset(registered_application_slugs())


def test_frame_sharing_hides_model_frame_and_rotates_colors() -> None:
    session = FrameSharingSession(_session().session_desc, device="cpu")
    session.init()
    model_generation_thread = session._ensure_thread_manager()._get_thread(0)
    assert isinstance(model_generation_thread, FrameSharingModelThread)

    results = [
        model_generation_thread.step(step, UserInputEvents([]))
        for step in (0, 10, 20, 30)
    ]
    colors = [
        tuple(
            result.output[0, :, 0, 0]
            .add(1.0)
            .mul(127.5)
            .round()
            .to(torch.int32)
            .tolist()
        )
        for result in results
    ]

    assert colors == [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0)]
    assert all(
        result.presentation_mode is PresentationMode.HIDE_PRESENTATION
        for result in results
    )


def test_message_demo_disables_model_presentation_and_updates_ui_by_message() -> None:
    session = MessageSession(_session().session_desc)
    session.init()
    ui_thread = session._ensure_thread_manager()._get_thread(1)
    model_thread = session._ensure_thread_manager()._get_thread(0)
    assert isinstance(ui_thread, MessageImGUIThread)
    assert isinstance(model_thread, MessageModelThread)
    events = UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(0),
                event_data=KeyboardUserInputEventData(
                    key="W", state=KeyboardInputState.Pressed
                ),
            )
        ]
    )

    result = model_thread.step(0, events)
    ui_thread._run_message_batch()

    assert result.presentation_mode is PresentationMode.DISABLE_PRESENTATION
    assert ui_thread.state.status == "W is Pressed"

    release_events = UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(1),
                event_data=KeyboardUserInputEventData(
                    key="W", state=KeyboardInputState.Released
                ),
            )
        ]
    )
    model_thread.step(1, release_events)
    ui_thread._run_message_batch()

    assert ui_thread.state.status == "W is not Pressed"


def test_init_registers_imgui_as_user_visible_thread() -> None:
    session = _session()

    session.init()
    threads = session._ensure_thread_manager()._freeze()

    assert list(threads) == [0, 1]
    assert threads[0].frequency == session.session_desc.frames_per_second_for_step
    assert threads[1].frequency == session.session_desc.frames_per_second_for_ui


def test_demo_builds_a_real_imgui_window_without_gpu_rendering() -> None:
    imgui = pytest.importorskip("imgui_bundle").imgui
    create_context = pytest.importorskip("slangpy.ui.imgui_bundle").create_imgui_context
    context = create_context(640, 480)
    thread = DemoImGUIThread(
        state=ImGUIDemoState(),
        frequency=60,
        output_layout=VideoTensorLayout.tchw,
        width=640,
        height=480,
    )
    try:
        imgui.set_current_context(context)
        imgui.get_io().set_ini_filename("")
        imgui.new_frame()
        thread.draw_ui(imgui, 0, UserInputEvents([]))
        imgui.render()
        draw_data = imgui.get_draw_data()

        assert draw_data.valid
        assert draw_data.total_vtx_count > 0
        assert draw_data.total_idx_count > 0
    finally:
        imgui.destroy_context(context)
