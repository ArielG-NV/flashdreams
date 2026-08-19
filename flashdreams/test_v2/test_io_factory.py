# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 I/O factory protocol."""

import pytest
import torch
from flashdreams.api_v2.input_handler import InputHandler
from flashdreams.api_v2.io_factory import IOFactory
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.core_v2.session_desc import SessionDesc
from flashdreams.core_v2.step_result import StepResult
from flashdreams.core_v2.user_input_event import (
    UserInputEvent,
    UnknownUserInputEventData,
    UserInputEventType,
)
from flashdreams.core_v2.user_input_events import UserInputEvents
from null_model import NULL_MODEL_CONFIG

pytestmark = pytest.mark.ci_cpu


class FakeInputHandler(InputHandler):
    """Return the latest available inputs."""

    def __init__(self) -> None:
        self._input: UserInputEvents | None = None

    def update_input(self, input: UserInputEvents) -> None:
        self._input = input

    def get_user_input_events(self) -> UserInputEvents:
        return self._input


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

    def __init__(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc

    def create_input_handler(self) -> FakeInputHandler:
        self.input_handler = FakeInputHandler()
        return self.input_handler

    def create_output_sink(self) -> OutputSink:
        self.output_sink = FakeOutputSink()
        return self.output_sink


def test_factory_gets_get_user_input_events_for_null_model() -> None:
    
    # Session layout matches the model's declared output layout.
    factory = FakeIOFactory(
        SessionDesc(
            output_layout=NULL_MODEL_CONFIG.output_layout,
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
        timestamp=1,
        event_type=UserInputEventType.UNKNOWN,
        event_data=UnknownUserInputEventData(data=2),
    )
    input_handler.update_input(
        UserInputEvents([unknown_input])
    )

    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()
    get_user_input_events = input_handler.get_user_input_events()
    
    assert get_user_input_events.get_events() == [unknown_input]
    event_data = get_user_input_events.get_events()[0].get_event_data()
    assert isinstance(event_data, UnknownUserInputEventData)
    
    output = pipeline.generate(0, cache, input=torch.tensor([[event_data.data]]))

    # model outputs layout bcthw, but in theory the model could output bctwh and we would require a swizzle operation to get to bcthw
    output_sink.open()
    output_sink.write(StepResult(
        step_index=0,
        output=output,
        frame_count=1,
        output_layout=NULL_MODEL_CONFIG.output_layout,
        metrics={},
    ))
    output_sink.close()

    assert event_data.data == 2
    assert output.shape == (1, 3, 1, 1, 1)
    assert output[0, 0, 0, 0, 0] == 2
