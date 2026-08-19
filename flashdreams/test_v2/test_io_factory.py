# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 I/O factory protocol."""

import pytest

from flashdreams.api_v2.input_handler import InputHandler
from flashdreams.api_v2.io_factory import IOFactory
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.core_v2.session_info import SessionInfo
from flashdreams.core_v2.step_result import StepResult
from flashdreams.core_v2.time_window import TimeWindow
from flashdreams.core_v2.user_input_event import (
    UserInputEvent,
    UserInputEventDataUnknown,
    UserInputEventType,
)
from flashdreams.core_v2.user_input_events import UserInputEvents
from flashdreams.core_v2.video_tensor import VideoTensorLayout
from null_model import NULL_MODEL_CONFIG

pytestmark = pytest.mark.ci_cpu


class FakeInputHandler(InputHandler):
    """Return the latest available inputs."""

    def __init__(self) -> None:
        self._is_open = False
        self._input: UserInputEvents = UserInputEvents(TimeWindow(0, 0), [])

    def update_input(self, input: UserInputEvents) -> None:
        self._input = input

    def open(self) -> None:
        self._is_open = True

    def current_inputs(self) -> UserInputEvents:
        assert self._is_open
        return self._input

    def close(self) -> None:
        self._is_open = False


class FakeOutputSink(OutputSink):
    """Record generated results for assertions."""

    def __init__(self) -> None:
        self.results: list[StepResult] = []
        self._is_open = False

    def open(self) -> None:
        self._is_open = True

    def write(self, result: StepResult) -> None:
        assert self._is_open
        self.results.append(result)

    def close(self) -> None:
        self._is_open = False


class FakeIOFactory(IOFactory):
    """Provide fake input and output edges for one session."""

    def __init__(self, session_info: SessionInfo) -> None:
        self.session_info = session_info

    def create_input_handler(self) -> FakeInputHandler:
        self.input_handler = FakeInputHandler()
        return self.input_handler

    def create_output_sink(self) -> OutputSink:
        self.output_sink = FakeOutputSink()
        return self.output_sink


def test_factory_gets_current_inputs_for_null_model() -> None:
    
    # Session outputs layout bcthw
    factory = FakeIOFactory(
        SessionInfo(
            output_layout=VideoTensorLayout.bcthw,
            frames_per_second_for_ui=1,
            frames_per_second_for_step=1,
            video_width=1,
            video_height=1,
        )
    )
    assert isinstance(factory, IOFactory)
    
    input_handler = factory.create_input_handler()
    assert isinstance(input_handler, InputHandler)

    output_sink = factory.create_output_sink()
    assert isinstance(output_sink, OutputSink)

    unknown_input = UserInputEvent(
        timestamp=0,
        event_type=UserInputEventType.UNKNOWN,
        event_data=UserInputEventDataUnknown(data=1),
    )
    input_handler.update_input(
        UserInputEvents(TimeWindow(0, 1), [unknown_input])
    )

    input_handler.open()
    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()
    current_inputs = input_handler.current_inputs()
    input_handler.close()

    output = pipeline.generate(0, cache, input=current_inputs)

    # model outputs layout bcthw, but in theory the model could output bctwh and we would require a swizzle operation to get to bcthw
    output_sink.open()
    output_sink.write(StepResult(
        step_index=0,
        output=output,
        frame_count=1,
        output_layout=VideoTensorLayout.bcthw,
        metrics={},
    ))
    output_sink.close()


    assert current_inputs.get_events() == [unknown_input]
    event_data = current_inputs.get_events()[0].get_event_data()
    assert isinstance(event_data, UserInputEventDataUnknown)
    assert event_data.data == 1
    assert output.shape == (1, 3, 1, 1, 1)
