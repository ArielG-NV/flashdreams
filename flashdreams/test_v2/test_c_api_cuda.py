# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA buffer, producer ordering, and callback lifetime checks for the C API."""

import ctypes as ct
import sys

import pytest
import torch

from flashdreams.runtime_v2.c_client_window import CClientWindow, _ClientWindowCallbacks
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = [
    pytest.mark.ci_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
@pytest.mark.parametrize("callback_fails", [False, True])
def test_cuda_delivery_orders_production_and_retains_pending_reads(
    source_device: str, callback_fails: bool
) -> None:
    """Read pixels asynchronously after write returns, including its error path."""
    driver = (
        ct.WinDLL("nvcuda.dll") if sys.platform == "win32" else ct.CDLL("libcuda.so.1")
    )
    copy = driver.cuMemcpyDtoHAsync_v2
    copy.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_size_t, ct.c_void_p]
    copy.restype = ct.c_int
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    outputs = []
    metadata = []

    def write(context, pointer):
        frame = pointer.contents
        host = torch.empty(
            (frame.frames, frame.height, frame.width, 3),
            dtype=torch.uint8,
            pin_memory=True,
        )
        outputs.append(host)
        # Leave a pending read beyond the callback and exercise allocator reuse
        # on subsequent writes. The C consumer queues on the supplied stream.
        torch.cuda._sleep(2_000_000)
        status = copy(
            host.data_ptr(), frame.device_pointer, frame.byte_count, frame.stream
        )
        metadata.append(
            (
                context,
                frame.device_index,
                frame.stream or 0,
                frame.byte_count,
                frame.step_index,
                status,
            )
        )
        return -1 if callback_fails else status

    callbacks = _ClientWindowCallbacks()
    callbacks.struct_size = ct.sizeof(callbacks)
    callbacks.abi_version = 2
    callbacks.user_data = 123
    callbacks.open = type(callbacks.open)(lambda *args: 0)
    callbacks.get_user_input_events = type(callbacks.get_user_input_events)(
        lambda *args: 0
    )
    callbacks.write = type(callbacks.write)(write)
    callbacks.close = type(callbacks.close)(lambda *args: None)
    window = CClientWindow(callbacks)
    window.open(SessionDesc(video_width=3, video_height=2))
    try:
        for step in range(8):
            with torch.cuda.stream(producer):
                source = torch.full((1, 3, 2, 3), -1.0, device=source_device)
                if source.is_cuda:
                    torch.cuda._sleep(2_000_000)
                source[:, step % 3].fill_(1.0)
                result = StepResult(
                    step_index=step,
                    output=source,
                    frame_count=1,
                    output_layout=VideoTensorLayout.tchw,
                )
            with torch.cuda.stream(consumer):
                if callback_fails:
                    with pytest.raises(RuntimeError, match="write callback failed"):
                        window.write(result)
                else:
                    window.write(result)
            del result, source
        consumer.synchronize()
    finally:
        consumer.synchronize()
        producer.synchronize()
        window.close()
    assert len(outputs) == 8
    for step, host in enumerate(outputs):
        expected = torch.zeros_like(host)
        expected[..., step % 3] = 255
        assert torch.equal(host, expected)
        assert metadata[step] == (123, device.index, consumer.cuda_stream, 18, step, 0)
