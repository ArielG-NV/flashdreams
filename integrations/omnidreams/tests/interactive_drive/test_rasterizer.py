# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import concurrent.futures
import contextlib
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from omnidreams.interactive_drive.rasterizer import (
    _LazyRasterFrame,
    _LoadedSceneData,
    _LudusConditionRasterizerImpl,
    _RenderedCameraFrames,
)


class _Event:
    def __init__(self) -> None:
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1


def _impl_for_render_chunk(*, use_cuda_frames: bool) -> _LudusConditionRasterizerImpl:
    impl = _LudusConditionRasterizerImpl.__new__(_LudusConditionRasterizerImpl)
    impl._scene_data = _LoadedSceneData(clipgt_scene=object(), scene_adapter=object())
    impl._scene_id = 7
    impl._selected_camera_name = "front"
    impl._all_camera_map = {"front": 0}
    impl._sensor_to_rig = {"front": torch.eye(4)}
    impl._bev = None
    impl._render_executor = None
    impl._bev_buffer_frames = 0
    impl._bev_frame_buffer = deque()
    impl._bev_stream = None
    impl._bev_camera_id = None
    impl._bev_sensor_to_rig = None
    impl._temp_dir = None
    impl._device = torch.device("cpu")
    impl._raster = SimpleNamespace(height=2, width=3)
    impl._use_cuda_frames = use_cuda_frames
    impl._to_ludus_camera_pose = lambda poses: poses

    def fake_render_one_camera(**kwargs):
        n_frames = int(kwargs["timestamps_batch"].shape[0])
        frames = torch.arange(n_frames * 2 * 3 * 3, dtype=torch.uint8).reshape(
            n_frames, 2, 3, 3
        )
        return _RenderedCameraFrames(frames_hwc_uint8=frames, ready_event=_Event())

    impl._render_one_camera = fake_render_one_camera
    return impl


def test_raster_chunk_uses_cuda_backed_frames_by_default() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)

    chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0),
        np.array([1, 2], dtype=np.int64),
    )

    assert callable(getattr(chunk.frames[0].rgb_host_uint8, "to_cuda_tensor", None))


def test_raster_chunk_can_disable_cuda_backed_frames() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=False)

    chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0),
        np.array([1, 2], dtype=np.int64),
    )

    first = chunk.frames[0].rgb_host_uint8
    assert isinstance(first, np.ndarray)
    assert not callable(getattr(first, "to_cuda_tensor", None))
    assert np.array_equal(first, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))


def test_render_query_metadata_stays_on_host(monkeypatch) -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    full_devices: list[object | None] = []
    render_args: tuple[object, ...] | None = None
    torch_full = torch.full

    def tracked_full(*args: object, **kwargs: object) -> torch.Tensor:
        full_devices.append(kwargs.get("device"))
        return torch_full(*args, **kwargs)

    class _Context:
        needs_vflip = False

        def render(self, *args: object, **kwargs: object) -> torch.Tensor:
            nonlocal render_args
            del kwargs
            render_args = args
            return torch.zeros((2, 2, 3, 4), dtype=torch.uint8)

    monkeypatch.setattr(torch, "full", tracked_full)
    impl.ctx = _Context()

    _LudusConditionRasterizerImpl._render_one_camera(
        impl,
        rig_poses=torch.eye(4).repeat(2, 1, 1),
        timestamps_batch=torch.tensor([1, 2], dtype=torch.int64),
        scene_id=7,
        camera_id=3,
        sensor_to_rig=torch.eye(4),
        camera_type=1,
        resolution=(2, 3),
    )

    assert full_devices == [None, None, None]
    assert render_args is not None
    metadata = render_args[:4]
    assert all(isinstance(tensor, torch.Tensor) for tensor in metadata)
    assert all(tensor.device.type == "cpu" for tensor in metadata)


def test_bev_publication_trails_rendering_by_steady_chunk_size() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    impl._bev = SimpleNamespace(enabled=True, height=2, width=3)
    impl._bev_camera_id = 1
    impl._bev_sensor_to_rig = torch.eye(4)
    impl._bev_buffer_frames = 8
    impl.reset_bev_buffer()
    submitted: list[concurrent.futures.Future[_RenderedCameraFrames]] = []

    class _DeferredExecutor:
        def submit(
            self, *args: object, **kwargs: object
        ) -> concurrent.futures.Future[_RenderedCameraFrames]:
            del args, kwargs
            future: concurrent.futures.Future[_RenderedCameraFrames] = (
                concurrent.futures.Future()
            )
            submitted.append(future)
            return future

    impl._render_executor = _DeferredExecutor()

    first_chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 5, axis=0),
        np.arange(5, dtype=np.int64),
    )
    assert all(
        not np.asarray(frame.bev_host_uint8).any() for frame in first_chunk.frames
    )
    assert not submitted[0].done()

    second_chunk = impl.render_chunk(
        np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 8, axis=0),
        np.arange(5, 13, dtype=np.int64),
    )
    assert all(
        not np.asarray(frame.bev_host_uint8).any() for frame in second_chunk.frames[:3]
    )

    rendered_first_chunk = torch.stack(
        [torch.full((2, 3, 3), value, dtype=torch.uint8) for value in range(1, 6)]
    )
    submitted[0].set_result(
        _RenderedCameraFrames(
            frames_hwc_uint8=rendered_first_chunk,
            ready_event=None,
        )
    )
    assert [
        int(np.asarray(frame.bev_host_uint8)[0, 0, 0])
        for frame in second_chunk.frames[3:]
    ] == [1, 2, 3, 4, 5]
    assert not submitted[1].done()


def test_reset_bev_buffer_drops_rendered_rollout_frames() -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    impl._bev = SimpleNamespace(enabled=True, height=2, width=3)
    impl._bev_buffer_frames = 2
    completed: concurrent.futures.Future[_RenderedCameraFrames] = (
        concurrent.futures.Future()
    )
    completed.set_result(
        _RenderedCameraFrames(
            frames_hwc_uint8=torch.ones((2, 2, 3, 3), dtype=torch.uint8),
            ready_event=None,
        )
    )
    impl._bev_frame_buffer.extend([(completed, 0), (completed, 1)])

    impl.reset_bev_buffer()

    assert len(impl._bev_frame_buffer) == 2
    assert all(
        not np.asarray(
            _LazyRasterFrame(None, index, rendered_frames_future=future)
        ).any()
        for future, index in impl._bev_frame_buffer
    )


def test_deferred_bev_waits_for_inputs_and_uses_dedicated_stream(monkeypatch) -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    calls: list[tuple[str, object]] = []
    input_ready_event = object()

    class _Stream:
        def wait_event(self, event: object) -> None:
            calls.append(("wait", event))

    bev_stream = _Stream()
    impl._bev_stream = bev_stream

    @contextlib.contextmanager
    def use_stream(stream: object):
        calls.append(("stream", stream))
        yield

    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    rendered = impl._render_bev(
        rig_poses=torch.eye(4).unsqueeze(0),
        timestamps_batch=torch.ones(1, dtype=torch.int64),
        scene_id=7,
        camera_id=1,
        sensor_to_rig=torch.eye(4),
        resolution=(2, 3),
        input_ready_event=input_ready_event,
    )

    assert rendered.frames_hwc_uint8.shape == (1, 2, 3, 3)
    assert calls == [("wait", input_ready_event), ("stream", bev_stream)]


def test_bev_input_event_records_on_current_stream(monkeypatch) -> None:
    impl = _impl_for_render_chunk(use_cuda_frames=True)
    impl._bev_stream = object()
    current_stream = object()

    class _EventWithRecord:
        def __init__(self) -> None:
            self.recorded_stream: object | None = None

        def record(self, stream: object) -> None:
            self.recorded_stream = stream

    event = _EventWithRecord()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current_stream)

    assert impl._record_bev_input_event() is event
    assert event.recorded_stream is current_stream
