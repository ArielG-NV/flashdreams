# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed workload settings for the Cosmos3 backend comparison."""

import subprocess
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
