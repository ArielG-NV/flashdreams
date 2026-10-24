# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AVIF sequence recording through the standalone libavif encoder."""

import os
import subprocess
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, TemporaryFile
from typing import BinaryIO, cast

import numpy as np
import numpy.typing as npt
from flashdreams_aom import decoder_path, encoder_path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor


class AvifClientWindow(IClientWindow):
    """Record lossy YUV 4:2:0 AV1 as an AVIF image sequence.

    Writes block when the encoder falls behind. Smaller presentations are padded
    on the right and bottom; growing the canvas migrates earlier frames.
    Close flushes the encoder and publishes the file. A single frame becomes
    a still image; sequences carry explicit frame durations and play once.
    libavif retains compressed samples until close, so memory grows with the
    encoded recording size.
    """

    def __init__(self, path: str | Path) -> None:
        """Configure the destination and locate the installed AOM encoder."""
        self._executable = encoder_path()
        decoder_path()
        self._path = Path(path)
        self._session_desc: SessionDesc | None = None
        self._presentation_size = (0, 0)
        self._canvas_size = (0, 0)
        self._process: subprocess.Popen[bytes] | None = None
        self._errors: BinaryIO | None = None
        self._staged: Path | None = None
        self._input_events: list[UserInputEvent] = []

    @property
    def path(self) -> Path:
        """Return the output path."""
        return self._path

    def get_user_input_events(self) -> UserInputEvents:
        """Return the events queued since the last call."""
        events, self._input_events = self._input_events, []
        return UserInputEvents(events)

    def put_user_input_event(self, event: UserInputEvent) -> None:
        self._input_events.append(event)

    def open(self, session_desc: SessionDesc) -> None:
        """Configure a recording without creating a file until frames arrive."""
        if self._session_desc is not None:
            raise RuntimeError("AvifClientWindow is already open.")
        self._session_desc = session_desc
        self._input_events = []
        self._presentation_size = (session_desc.video_width, session_desc.video_height)
        self._canvas_size = self._presentation_size

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Resize the presentation on a growing, top-left-aligned canvas.

        Raises:
            RuntimeError: The window is not open or canvas migration fails.
            ValueError: Dimensions are invalid.
        """
        if self._session_desc is None:
            raise RuntimeError("AvifClientWindow.open() must run before resizing.")
        if any(dimension <= 0 for dimension in new_window_size):
            raise ValueError("Presentation width and height must be > 0.")
        canvas_size = (
            max(self._canvas_size[0], new_window_size[0]),
            max(self._canvas_size[1], new_window_size[1]),
        )
        if canvas_size != self._canvas_size:
            if self._process is not None:
                self._expand_canvas(canvas_size)
            else:
                self._canvas_size = canvas_size
        self._presentation_size = new_window_size

    def write(self, result: StepResult) -> None:
        """Encode incoming frames synchronously, releasing them after the write."""
        desc = self._session_desc
        if desc is None:
            raise RuntimeError("AvifClientWindow.open() must run before write().")
        frames = result_to_rgb24_tensor(result, desc, self._presentation_size)
        if not len(frames):
            return
        try:
            self._process or self._start(desc)
            for frame in frames.cpu():
                self._write_planes(_rgb_to_yuv420(frame.numpy()))
            assert self._process is not None and self._process.stdin is not None
            self._process.stdin.flush()
        except BrokenPipeError as error:
            assert self._process is not None
            self._process.wait()
            message = self._failure()
            self._discard()
            raise RuntimeError(message) from error
        except BaseException:
            self._discard()
            raise

    def close(self) -> None:
        """Flush delayed packets and atomically publish a successful recording."""
        self._session_desc = None
        process = self._process
        if process is None:
            return
        try:
            self._finish()
            assert self._staged is not None
            self._staged.replace(self._path)
        finally:
            self._discard()

    def _finish(self) -> None:
        """Flush the active encoder and check its exit status."""
        process = self._process
        assert process is not None and process.stdin is not None
        try:
            process.stdin.close()
        except OSError:
            if process.wait() == 0:
                raise
        if process.wait() != 0:
            raise RuntimeError(self._failure())

    def _write_planes(self, planes: tuple[npt.NDArray[np.uint8], ...]) -> None:
        """Pad one YUV 4:2:0 frame to the canvas and send it to AOM."""
        width, height = self._canvas_size
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(b"FRAME\n")
        for index, plane in enumerate(planes):
            shape = (
                (height, width) if index == 0 else ((height + 1) // 2, (width + 1) // 2)
            )
            if plane.shape != shape:
                plane = np.pad(
                    plane,
                    ((0, shape[0] - plane.shape[0]), (0, shape[1] - plane.shape[1])),
                    constant_values=16 if index == 0 else 128,
                )
            self._process.stdin.write(plane.tobytes())

    def _expand_canvas(self, size: tuple[int, int]) -> None:
        """Decode and re-encode earlier frames into a larger recording."""
        executable = decoder_path()
        source = self._staged
        desc = self._session_desc
        assert source is not None and desc is not None
        width, height = self._canvas_size
        luma_size = width * height
        chroma_size = ((width + 1) // 2) * ((height + 1) // 2)
        frame_size = luma_size + 2 * chroma_size
        try:
            # ponytail: expansion re-encodes history with quality loss; use segments for frequent resizing.
            self._finish()
            self._process = None
            assert self._errors is not None
            self._errors.close()
            self._errors = None
            self._staged = None
            self._canvas_size = size
            self._start(desc)
            # ponytail: avifdec exports all frames to disk; use a streaming decoder for frequent resizing.
            with (
                TemporaryDirectory(dir=source.parent) as directory,
                TemporaryFile() as errors,
            ):
                decoder = subprocess.run(
                    [
                        str(executable),
                        "--jobs",
                        "8",
                        "--index",
                        "all",
                        str(Path("..") / source.name),
                        "frame.y4m",
                    ],
                    stdout=errors,
                    stderr=errors,
                    cwd=directory,
                )
                if decoder.returncode != 0:
                    errors.seek(0, os.SEEK_END)
                    errors.seek(max(0, errors.tell() - 8192))
                    detail = errors.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"AVIF failed while resizing: {detail}")
                frames = sorted(Path(directory).glob("frame-*.y4m"))
                if not frames:
                    raise RuntimeError("AVIF decoder returned no frames.")
                for frame in frames:
                    with frame.open("rb") as decoded:
                        if (
                            not decoded.readline().startswith(b"YUV4MPEG2 ")
                            or decoded.readline() != b"FRAME\n"
                        ):
                            raise RuntimeError("AVIF decoder returned invalid Y4M.")
                        data = decoded.read(frame_size + 1)
                    if len(data) != frame_size:
                        raise RuntimeError("AVIF decoder returned an incomplete frame.")
                    samples = np.frombuffer(data, dtype=np.uint8)
                    self._write_planes(
                        (
                            samples[:luma_size].reshape(height, width),
                            samples[luma_size : luma_size + chroma_size].reshape(
                                (height + 1) // 2, (width + 1) // 2
                            ),
                            samples[luma_size + chroma_size :].reshape(
                                (height + 1) // 2, (width + 1) // 2
                            ),
                        )
                    )
                    frame.unlink()
        except BrokenPipeError as error:
            message = self._failure()
            self._discard()
            raise RuntimeError(message) from error
        except BaseException:
            self._discard()
            raise
        finally:
            source.unlink(missing_ok=True)

    def _start(self, desc: SessionDesc) -> subprocess.Popen[bytes]:
        """Start the bundled encoder with explicit frame timing and YUV input."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self._path.parent, suffix=".avif", delete=False
        ) as temporary:
            self._staged = Path(temporary.name)
        # File-backed diagnostics avoid both a reader thread and a full stderr pipe.
        self._errors = cast(BinaryIO, TemporaryFile())
        width, height = self._canvas_size
        fps = Fraction(str(desc.frames_per_second_for_step)).limit_denominator(
            1_000_000
        )
        # ponytail: libavif buffers compressed samples until EOF; use fragmented MP4 for bounded memory.
        self._process = subprocess.Popen(
            [
                str(self._executable),
                "--stdin",
                "--codec",
                "aom",
                "--speed",
                "10",
                "--qcolor",
                "80",
                "--advanced",
                "end-usage=q",
                "--cicp",
                "1/13/1",
                "--jobs",
                str(min(8, os.cpu_count() or 1)),
                "--timescale",
                str(fps.numerator),
                "--duration",
                str(fps.denominator),
                "--repetition-count",
                "0",
                "--output",
                self._staged.name,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._errors,
            cwd=self._staged.parent,
        )
        assert self._process.stdin is not None
        self._process.stdin.write(
            f"YUV4MPEG2 W{width} H{height} F{fps.numerator}:{fps.denominator} "
            "Ip A1:1 C420jpeg XCOLORRANGE=LIMITED\n".encode("ascii")
        )
        return self._process

    def _failure(self) -> str:
        """Return a bounded diagnostic from the encoder's error file."""
        assert self._errors is not None
        self._errors.seek(0, os.SEEK_END)
        self._errors.seek(max(0, self._errors.tell() - 8192))
        detail = (
            self._errors.read().decode("utf-8", errors="replace").strip() or "no output"
        )
        return f"AVIF failed while writing {self._path}: " + f"{detail}"

    def _discard(self) -> None:
        """Release native resources and remove unpublished output."""
        self._session_desc = None
        process, self._process = self._process, None
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except OSError:
                    # Windows can report EINVAL when closing an exited child's pipe.
                    pass
                finally:
                    process.wait()
        finally:
            if self._errors is not None:
                self._errors.close()
                self._errors = None
            if self._staged is not None:
                self._staged.unlink(missing_ok=True)
                self._staged = None


def _rgb_to_yuv420(frame: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], ...]:
    """Convert packed RGB to limited-range BT.709 with averaged 2x2 chroma."""
    yuv = frame.astype(np.float32) @ np.array(
        [
            [0.182586, -0.100644, 0.439216],
            [0.614231, -0.338572, -0.398942],
            [0.062007, 0.439216, -0.040274],
        ],
        dtype=np.float32,
    )
    yuv += np.array([16, 128, 128], dtype=np.float32)
    chroma = np.pad(
        yuv[:, :, 1:],
        ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)),
        mode="edge",
    )
    chroma = chroma.reshape(chroma.shape[0] // 2, 2, chroma.shape[1] // 2, 2, 2).mean(
        axis=(1, 3)
    )
    return (
        yuv[:, :, 0].round().clip(16, 235).astype(np.uint8),
        chroma[:, :, 0].round().clip(16, 240).astype(np.uint8),
        chroma[:, :, 1].round().clip(16, 240).astype(np.uint8),
    )
