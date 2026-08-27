# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC smoke coverage for the runtime-v2 model process."""

import os

import pytest
import torch

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


class _CudaModelLoop(IModelLoop[SessionDesc]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        del events
        output = torch.tensor(
            [1.0, 2.0, 3.0], device="cuda", dtype=torch.float32
        ).reshape(1, 3, 1, 1)
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                metrics={"model_process_id": os.getpid()},
            )
        ]


class _CudaSession(ISession):
    def __init__(self) -> None:
        self._desc = SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
            frames_per_second_for_ui=120,
            frames_per_second_for_step=120,
            video_width=1,
            video_height=1,
        )

    @property
    def session_desc(self) -> SessionDesc:
        return self._desc

    def init(self) -> None:
        self.register_model_loop(_CudaModelLoop, state=self._desc)


class _Window(IClientWindow):
    def __init__(self) -> None:
        self.values: list[float] | None = None

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])

    def write(self, result: StepResult) -> None:
        self.values = result.output.flatten().cpu().tolist()

    def close(self) -> None:
        return


class _MetricsSink:
    def __init__(self) -> None:
        self.was_cuda = False
        self.model_process_id: int | None = None

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc

    def write(self, result: StepResult) -> None:
        self.was_cuda = result.output.is_cuda
        self.model_process_id = int(result.metrics["model_process_id"])

    def close(self) -> None:
        return


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_result_crosses_process_without_host_materialization() -> None:
    window = _Window()
    metrics = _MetricsSink()

    run_session(_CudaSession(), window, metrics_output_sink=metrics, steps=1)

    assert metrics.was_cuda
    assert metrics.model_process_id != os.getpid()
    assert window.values == [1.0, 2.0, 3.0]
