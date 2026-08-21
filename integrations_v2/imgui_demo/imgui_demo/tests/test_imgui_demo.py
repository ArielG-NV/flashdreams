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
from imgui_demo.app import DemoImGUIThread, ImGUIDemoSession, ImGUIDemoState

from flashdreams.runtime_v2.session_desc import SessionDesc
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


def test_main_generation_result_is_disabled() -> None:
    session = _session()

    result = session.step(3, UserInputEvents([]))

    assert result.step_index == 3
    assert result.disabled
    assert result.output.shape == (1, 3, 24, 32)


def test_init_registers_imgui_as_auxiliary_thread() -> None:
    session = _session()

    session.init()
    threads = session._freeze_thread_registry()

    assert list(threads) == [1]
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
