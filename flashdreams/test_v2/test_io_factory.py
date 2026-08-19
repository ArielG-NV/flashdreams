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
    NumeralKeypadUserInputEventData,
)
from flashdreams.core_v2.user_input_events import UserInputEvents
from null_model import NULL_MODEL_CONFIG

pytestmark = pytest.mark.ci_cpu


class FakeInputHandler(InputHandler):
    """Return the latest available inputs."""

    def __init__(self) -> None:
        self._input: UserInputEvents | None = None

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

    def update_input_events(self, input: UserInputEvents) -> None:
        self.input_handler._input = input

def test_factory_gets_get_user_input_events_for_null_model() -> None:
    
    # Session layout desc + Factory + InputHandler + OutputSink setup
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
    ## Assume the sink takes time to open due to startup time for backend
    output_sink.open()

    # Pipeline setup
    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()

    current_timestamp = -100
    current_step_index = -1
    test_event_data = 2
    while current_timestamp < 1000:
        current_step_index += 1
        current_timestamp += 100
        numeral_keypad_input = UserInputEvent(
            timestamp=current_timestamp,
            event_data=NumeralKeypadUserInputEventData(value=test_event_data),
        )

        # This is the presentation backend updating user-inputs handled by the
        # input handler.
        factory.update_input_events(
            UserInputEvents([numeral_keypad_input])
        )

        # This is the input handler getting the user-inputs to send to our `step`/`ui_step` loops
        get_user_input_events = input_handler.get_user_input_events()        
        assert get_user_input_events.get_events() == [numeral_keypad_input]
        event_data = get_user_input_events.get_events()[0].get_event_data()

        # This is inside our `step` loop.
        output = pipeline.generate(current_step_index, cache, input=torch.tensor([[event_data.value]]))
        ## Note: model output is in bcthw layout, but in theory the model could output bctwh and we would require a swizzle operation to get to bcthw
        output_sink.write(StepResult(
            step_index=current_step_index,
            output=output,
            frame_count=1,
            output_layout=NULL_MODEL_CONFIG.output_layout,
            metrics={},
        ))

        assert numeral_keypad_input.get_event_data().__hash__() == NumeralKeypadUserInputEventData.__hash__()
        assert event_data.value == test_event_data
        assert output.shape == (1, 3, 1, 1, 1)
        assert output[0, 0, 0, 0, 0].item() == current_step_index + test_event_data
    output_sink.close()
