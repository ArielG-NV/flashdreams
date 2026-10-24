# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched Cosmos3 workload settings and validated video export."""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/three-seconds"
REVISIONS = {
    "bimanual": "be4e2bc243f9f4a0b213eeebd1308cfbac8d9d7e",
    "auto": "c3fa1378e93710242501a6288d7c9e7c9084a580",
}
SHAPES = {"bimanual": (640, 352), "auto": (832, 480)}
CHUNKS = {"bimanual": 2, "auto": 3}
LATENTS = {"bimanual": [48, 25, 22, 40], "auto": [48, 25, 30, 52]}
HISTORY = {"bimanual": 25, "auto": 4}
TIMESTEPS = [1.0, 0.9375, 5 / 6, 0.625]
WORK_FRAMES, GENERATED_FRAMES, FPS, SEED = 97, 96, 30, 0
WARMUP, TRIALS = 2, 5


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def save_video(pixels, out: Path) -> dict:
    """Save and fully decode a 97-frame RGB clip, returning artifact metadata."""
    if str(pixels.dtype) != "uint8" or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("Expected uint8 THWC RGB video")
    if len(pixels) != WORK_FRAMES:
        raise ValueError("Expected initial frame plus 96 generated frames")
    path = out / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{pixels.shape[2]}x{pixels.shape[1]}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        input=pixels.tobytes(),
        check=True,
        capture_output=True,
    )
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    if (
        int(stream["nb_read_frames"]),
        int(stream["height"]),
        int(stream["width"]),
        Fraction(stream["avg_frame_rate"]),
    ) != (*pixels.shape[:3], FPS):
        raise ValueError("Encoded video differs from generated pixels")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return {
        "path": path.relative_to(HERE).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": len(pixels),
        "fps": FPS,
        "width": pixels.shape[2],
        "height": pixels.shape[1],
    }
