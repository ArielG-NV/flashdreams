# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for native-window output."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from flashdreams.runtime import UserInputEvent
from flashdreams.runtime.demo import (
    NativeWindowOutputSpec,
    RealtimeEventInputSource,
    SessionInfo,
)
from flashdreams.runtime.types import StepResult
from flashdreams.serving.native_window.services import (
    NativeFrameQueue,
    NativeWindowInputSource,
    NativeWindowOutputSink,
)

pytestmark = pytest.mark.ci_cpu


def _result(value: int) -> StepResult:
    return StepResult.from_video_chunk(
        step_index=value,
        video_chunk=torch.full((2, 3, 2, 3), value, dtype=torch.uint8),
        layout="tchw",
    )


def test_native_source_uses_generic_events() -> None:
    source = NativeWindowInputSource(fps=20)
    assert isinstance(source, RealtimeEventInputSource)

    source.record_key(event="keydown", key="w", timestamp_s=0.1)
    source.record_user_event(
        UserInputEvent(
            timestamp_s=0.2,
            event_type="text_event",
            payload={"event_id": "rain", "state": "trigger"},
        )
    )
    assert [event.event_type for event in source._events_for_window(0.0, 1.0)] == [
        "key_down",
        "text_event",
    ]


def test_native_output_spec_defaults_and_validation() -> None:
    output = NativeWindowOutputSpec()
    assert output.mode == "native-window"
    assert output.max_queued_chunks == 2
    with pytest.raises(ValueError, match="dimensions"):
        NativeWindowOutputSpec(video_width=0)


def test_native_queue_preserves_frames_and_drops_stale_chunks() -> None:
    queue = NativeFrameQueue(max_chunks=1)
    dropped, queued = queue.publish(_result(1))
    assert not dropped and queued == 2

    dropped, queued = queue.publish(_result(2))
    assert dropped and queued == 2
    frame = queue.pop()
    assert frame is not None
    assert cast(Any, frame).to_numpy()[0, 0].tolist() == [2, 2, 2]


def test_native_sink_clears_between_generations() -> None:
    queue = NativeFrameQueue(max_chunks=2)
    sink = NativeWindowOutputSink(queue=queue, fps=10)
    sink.open(SessionInfo())
    sink.write(_result(1))

    sink.begin_generation(1)
    assert queue.pop() is None
    assert sink.write(_result(2)).backpressure_s == pytest.approx(0.2)
    assert sink.close() == ()
    assert queue.closed
