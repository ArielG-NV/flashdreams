# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 presentation surface protocol."""

import pytest
import torch
from flashdreams.api_v2.input_handler import InputHandler
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.api_v2.presentation_surface import IPresentationSurface
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    UserInputEvent,
    NumeralKeypadUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from null_model import NULL_MODEL_CONFIG

pytestmark = pytest.mark.ci_cpu


class FakePresentationSurface(IPresentationSurface):
    """Provide fake presentation input and output for one session."""
    def __init__(self, session_desc: SessionDesc) -> None:
        super().__init__(session_desc)
        self._input: UserInputEvents | None = None
        self.results: list[StepResult] = []

        # Only applies to writing to output; reading from input will just not produce "new"
        # results if closed, it does not imply the input handler contains invalid data.
        self._is_open = False

    def get_user_input_events(self) -> UserInputEvents:
        return self._input

    def open(self) -> None:
        self._is_open = True

    def write(self, result: StepResult) -> None:
        assert self._is_open
        self.results.append(result)

    def close(self) -> None:
        self._is_open = False

    def update_input_events(self, input: UserInputEvents) -> None:
        self._input = input


def test_presentation_surface_for_null_model() -> None:
    
    # Session layout desc + Factory + InputHandler + OutputSink setup
    presentation_surface = FakePresentationSurface(
        SessionDesc(
            output_layout=NULL_MODEL_CONFIG.output_layout,
            frames_per_second_for_ui=1,
            frames_per_second_for_step=1,
            video_width=1,
            video_height=1,
        )
    )
    assert isinstance(presentation_surface, IPresentationSurface)
    assert isinstance(presentation_surface, InputHandler)
    assert isinstance(presentation_surface, OutputSink)
    ## Assume the sink takes time to open due to startup time for backend
    presentation_surface.open()

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
        presentation_surface.update_input_events(
            UserInputEvents([numeral_keypad_input])
        )

        # This is the input handler getting the user-inputs to send to our `step`/`ui_step` loops
        get_user_input_events = presentation_surface.get_user_input_events()
        assert get_user_input_events.get_events() == [numeral_keypad_input]
        event_data = get_user_input_events.get_events()[0].get_event_data()

        # This is inside our `step` loop.
        output = pipeline.generate(current_step_index, cache, input=torch.tensor([[event_data.value]]))
        ## Note: model output is in bcthw layout, but in theory the model could output bctwh and we would require a swizzle operation to get to bcthw
        presentation_surface.write(StepResult(
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
    presentation_surface.close()
