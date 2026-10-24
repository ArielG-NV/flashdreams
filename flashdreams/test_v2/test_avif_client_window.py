# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for AVIF recordings and transactional failure cleanup."""

import argparse
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

import flashdreams_aom as aom
import numpy as np
import pytest
import torch

from flashdreams.runtime_v2.avif_client_window import AvifClientWindow, _rgb_to_yuv420
from flashdreams.runtime_v2.client_window_factory import (
    add_client_window_arguments,
    client_window_mode,
)
from flashdreams.runtime_v2.selected_file import (
    FileSelectionRequest,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import SelectedFilesUserInputEvent
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def test_avif_delivers_file_selector_events_once(tmp_path: Path) -> None:
    window = AvifClientWindow(tmp_path / "unused.avif")
    window.open(SessionDesc())
    window.request_file_selector(
        FileSelectionRequest(request_id="input", initial_path="")
    )
    events = window.get_user_input_events().get_events()
    assert len(events) == 1
    assert isinstance(events[0], SelectedFilesUserInputEvent)
    assert events[0].request_id == "input"
    assert events[0].status is SelectedFilesStatus.UNAVAILABLE
    assert window.get_user_input_events().get_events() == []
    window.close()


def test_yuv420_color_conversion_and_odd_dimensions() -> None:
    colors = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], dtype=np.uint8)
    y, u, v = _rgb_to_yuv420(colors)
    np.testing.assert_array_equal(y, [[16, 235, 63]])
    np.testing.assert_array_equal(u, [[128, 102]])
    np.testing.assert_array_equal(v, [[128, 240]])
    checker = np.array(
        [[[255, 0, 0], [0, 0, 255]], [[0, 0, 255], [255, 0, 0]]], dtype=np.uint8
    )
    _, u, v = _rgb_to_yuv420(checker)
    np.testing.assert_array_equal(u, [[171]])
    np.testing.assert_array_equal(v, [[179]])


def _boxes(data: bytes):
    """Read the container boxes needed to check AVIF sequence timing."""
    position = 0
    while position < len(data):
        size, tag = struct.unpack_from(">I4s", data, position)
        assert size >= 8 and position + size <= len(data)
        payload = data[position + 8 : position + size]
        yield tag, payload
        if tag in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
            yield from _boxes(payload)
        position += size


def _assert_timing(path: Path, frame_count: int, fps: float) -> None:
    boxes = dict(_boxes(path.read_bytes()))
    assert b"avif" in boxes[b"ftyp"]
    if frame_count == 1:
        assert b"moov" not in boxes  # libavif writes one frame as a still image.
        return
    assert b"avis" in boxes[b"ftyp"]
    header = boxes[b"mdhd"]
    assert header[0] == 1
    timescale, duration = struct.unpack_from(">IQ", header, 20)
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    assert timescale == rate.numerator
    assert duration == frame_count * rate.denominator
    # A single playback has the same duration in the movie and media headers.
    assert struct.unpack_from(">IQ", boxes[b"mvhd"], 20) == (timescale, duration)
    timing = boxes[b"stts"]
    entries = struct.unpack_from(">I", timing, 4)[0]
    durations = []
    for index in range(entries):
        count, delta = struct.unpack_from(">II", timing, 8 + index * 8)
        durations.extend([delta] * count)
    assert durations == [rate.denominator] * frame_count


def _decode(path: Path) -> np.ndarray:
    """Decode all frames without relying on system codecs or FFmpeg."""
    with TemporaryDirectory(dir=path.parent) as directory:
        subprocess.run(
            [
                str(aom.decoder_path()),
                "--jobs",
                "2",
                "--index",
                "all",
                str(Path("..") / path.name),
                "frame.y4m",
            ],
            cwd=directory,
            check=True,
            capture_output=True,
        )
        frames = []
        for frame in sorted(Path(directory).glob("frame-*.y4m")):
            header, marker, data = frame.read_bytes().split(b"\n", 2)
            assert header.startswith(b"YUV4MPEG2 ") and marker == b"FRAME"
            frames.append(np.frombuffer(data, dtype=np.uint8))
        return np.stack(frames)


def test_avif_needs_an_output_path() -> None:
    parser = argparse.ArgumentParser()
    add_client_window_arguments(parser)
    with pytest.raises(ValueError, match="--output-path"):
        client_window_mode("avif").check_arguments(
            parser.parse_args(["--mode", "avif"])
        )


def test_missing_native_dependency_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flashdreams_aom", None)
    with pytest.raises(
        RuntimeError, match="native dependency flashdreams-aom is missing"
    ):
        AvifClientWindow("unused.avif")


@pytest.mark.parametrize("missing", ["avifenc", "avifdec"])
def test_missing_native_binary_fails_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(aom, "__file__", str(tmp_path / "__init__.py"))
    suffix = ".exe" if sys.platform == "win32" else ""
    present = tmp_path / (("avifdec" if missing == "avifenc" else "avifenc") + suffix)
    present.touch()
    present.chmod(0o755)
    with pytest.raises(RuntimeError) as error:
        AvifClientWindow(tmp_path / "unused.avif")
    message = str(error.value)
    assert str(tmp_path / (missing + suffix)) in message
    assert "uv sync --reinstall-package flashdreams-aom" in message
    assert "C/C++" in message and "YASM" in message


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("frame_count", [1, 96])
@pytest.mark.parametrize("fps", [24.0, 30000 / 1001])
def test_streaming_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
    frame_count: int,
    fps: float,
) -> None:
    if fail:
        # Python rejects avifenc's arguments and exits, exercising a real broken pipe.
        monkeypatch.setitem(
            sys.modules,
            "flashdreams_aom",
            SimpleNamespace(
                encoder_path=lambda: Path(sys.executable),
                decoder_path=lambda: Path(sys.executable),
            ),
        )
    monkeypatch.setenv("PATH", "")
    tmp_path = tmp_path / "recording ü"
    tmp_path.mkdir()
    path = tmp_path / "clip.avif"
    path.write_bytes(b"original")
    window = AvifClientWindow(path)
    window.close()
    assert window.get_user_input_events().get_events() == []
    window.open(
        SessionDesc(
            video_width=16,
            video_height=8,
            # Exercise fractional rates supported at runtime despite the int annotation.
            frames_per_second_for_step=cast(int, fps),
        )
    )
    with pytest.raises(RuntimeError, match="already open"):
        window.open(SessionDesc())
    with pytest.raises(ValueError, match="> 0"):
        window.request_new_window_size((0, 8))
    window.request_new_window_size((19, 9))
    # Enough unpredictable data to exceed file/pipe buffers before close.
    rgb = torch.randint(0, 256, (frame_count, 3, 9, 19), dtype=torch.uint8)
    result = StepResult(
        step_index=0,
        output=rgb,
        frame_count=frame_count,
        output_layout=VideoTensorLayout.tchw,
    )

    def record() -> None:
        window.write(result)
        assert path.read_bytes() == b"original"
        window.request_new_window_size((19, 9))
        window.close()

    if fail:
        with pytest.raises(RuntimeError, match="AVIF failed"):
            record()
        assert path.read_bytes() == b"original"
    else:
        record()
        _assert_timing(path, frame_count, fps)
        yuv = _decode(path)
        assert len(yuv) == frame_count
        expected = np.stack(
            [
                np.concatenate(
                    [
                        plane.ravel()
                        for plane in _rgb_to_yuv420(frame.permute(1, 2, 0).numpy())
                    ]
                )
                for frame in rgb
            ]
        )
        assert np.abs(yuv.astype(float) - expected).mean() < 4
    window.close()
    assert window._process is None and window._errors is None and window._staged is None
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(RuntimeError, match="before write"):
        window.write(result)
    with pytest.raises(RuntimeError, match="before resizing"):
        window.request_new_window_size((8, 8))
    window.open(SessionDesc())
    window.close()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("fail_migration", [False, True])
def test_resize_preserves_frames_and_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_migration: bool
) -> None:
    monkeypatch.setenv("PATH", "")
    path = tmp_path / "resized.avif"
    path.write_bytes(b"original")
    window = AvifClientWindow(path)
    window.open(
        SessionDesc(video_width=16, video_height=8, frames_per_second_for_step=24)
    )
    expected = []
    sizes = [(16, 8), (7, 5), (21, 9), (9, 13), (6, 4)]
    for index, (width, height) in enumerate(sizes):
        if fail_migration and index == 2:
            monkeypatch.setattr(aom, "decoder_path", lambda: Path(sys.executable))
            with pytest.raises(RuntimeError, match="AVIF failed while resizing"):
                window.request_new_window_size((width, height))
            window.close()
            assert path.read_bytes() == b"original"
            assert list(tmp_path.iterdir()) == [path]
            assert window._process is None and window._errors is None
            return
        window.request_new_window_size((width, height))
        rgb = torch.tensor([31 + index, 129 + index, 245 - index], dtype=torch.uint8)
        rgb = rgb[:, None, None].expand(3, 8, 16).unsqueeze(0)
        if index == 0:
            rgb = torch.randint(0, 256, (1, 3, 8, 16), dtype=torch.uint8)
        window.write(
            StepResult(
                step_index=index,
                output=rgb,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        )
        frame = rgb[0].permute(1, 2, 0).numpy()
        if index:
            frame = np.broadcast_to(frame[:1, :1], (height, width, 3))
        planes = _rgb_to_yuv420(frame)
        expected.append(
            np.concatenate(
                [
                    np.pad(
                        plane,
                        (
                            (0, target[0] - plane.shape[0]),
                            (0, target[1] - plane.shape[1]),
                        ),
                        constant_values=16 if channel == 0 else 128,
                    ).ravel()
                    for channel, (plane, target) in enumerate(
                        zip(planes, [(13, 21), (7, 11), (7, 11)], strict=True)
                    )
                ]
            )
        )
        assert path.read_bytes() == b"original"
    window.close()
    yuv = _decode(path)
    assert np.abs(yuv.astype(float) - np.stack(expected)).mean() < 4
    _assert_timing(path, len(sizes), 24)
    assert list(tmp_path.iterdir()) == [path]
