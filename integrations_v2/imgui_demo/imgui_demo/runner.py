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

"""Shared WebRTC launcher for the Dear ImGui demos."""

import argparse
from collections.abc import Callable, Sequence

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.client_window_factory import create_client_window
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

_MODEL_GENERATION_FPS = 30
"""Fixed model-generation rate shared by the demos."""


def _parse_args(
    commandline_args: Sequence[str] | None,
    *,
    program: str,
    description: str,
) -> argparse.Namespace:
    """Parse shared WebRTC runtime arguments."""
    parser = argparse.ArgumentParser(prog=program, description=description)
    parser.add_argument("--mode", choices=("webrtc",), default="webrtc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60, help="UI frames per second.")
    return parser.parse_args(commandline_args)


def _session_desc(args: argparse.Namespace) -> SessionDesc:
    """Build the session description shared by every demo."""
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=args.fps,
        frames_per_second_for_step=_MODEL_GENERATION_FPS,
        video_width=args.width,
        video_height=args.height,
    )


def run_demo(
    create_app: Callable[[], IApplication],
    commandline_args: Sequence[str] | None,
    *,
    program: str,
    description: str,
) -> int:
    """Serve one ImGui demo until the browser disconnects."""
    args = _parse_args(
        commandline_args,
        program=program,
        description=description,
    )
    window = create_client_window(args)
    if isinstance(window, WebRTCClientWindow):
        print(f"Open {window.server.url} in a browser.", flush=True)
    try:
        ApplicationRunner(create_app(), window).run(_session_desc(args))
    except KeyboardInterrupt:
        return 130
    finally:
        window.close()
    return 0
