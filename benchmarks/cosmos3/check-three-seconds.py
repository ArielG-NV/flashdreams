# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for matched workloads, results and video export."""

import importlib.util
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark import (
    CHUNKS,
    HERE,
    HISTORY,
    LATENTS,
    REVISIONS,
    SHAPES,
    TIMESTEPS,
    save_video,
)

spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).with_name("report-three-seconds.py")
)
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
rows: list[dict[str, Any]] = [
    {
        "video": {
            "path": f"results/three-seconds/{model}/{backend}/output.mp4",
            "sha256": "0" * 64,
            "frames": 97,
            "fps": 30,
            "width": SHAPES[model][0],
            "height": SHAPES[model][1],
        },
        "status": "passed",
        "backend": backend,
        "gpu": "NVIDIA test GPU",
        "width": SHAPES[model][0],
        "height": SHAPES[model][1],
        "work_frames": 97,
        "generated_frames": 96,
        "steps": 4,
        "latent_shape": LATENTS[model],
        "history_capacity": HISTORY[model],
        "fps": 30,
        "seed": 0,
        "guidance_scale": 1.0,
        "decoder": "Wan 2.2 full VAE",
        "dtype": "bfloat16",
        "timesteps": TIMESTEPS,
        "warmup_requests": 2,
        "startup_s": 12.345,
        "trials_s": [2.0] * 5,
        "checkpoint_revision": revision,
        "latent_frames_per_chunk": CHUNKS[model],
        "input_sha256": "same",
        "prompt_sha256": "same",
        "actions_sha256": "same",
        "domain_ids": [2] * 96,
        "history_frames": 3,
        "vae_sha256": "same",
        "hostname": "PRIVATE_HOST",
        "cwd": "/private/directory",
        "username": "PRIVATE_USER",
    }
    for model, revision in REVISIONS.items()
    for backend in ("flashdreams", "vllm")
]
expected = report.format_results(rows)
assert expected.count("[Watch](results/three-seconds/") == 4
assert expected.count("| 48.00 | 2.00 | 2.00 |") == 4
assert "48 × 25 × 22 × 40" in expected
assert "48 × 25 × 30 × 52" in expected
assert "Latents/chunk" in expected and "Denoise steps/chunk" in expected
assert "Startup (s)" in expected
assert expected.count("| 12.35 | 48.00 |") == 4
assert "PRIVATE_HOST" not in expected and "PRIVATE_USER" not in expected
varied = deepcopy(rows)
varied[0]["trials_s"] = [1.0, 2.0, 3.0, 4.0, 10.0]
assert "| 24.00 | 3.00 | 10.00 |" in report.format_results(varied)

for index, key, value in (
    (0, "steps", 2),
    (0, "latent_shape", [48, 25, 30, 52]),
    (0, "history_capacity", 33),
    (1, "width", 832),
    (1, "height", 480),
    (1, "latent_frames_per_chunk", 4),
    (0, "work_frames", 90),
    (0, "generated_frames", 97),
    (0, "fps", 24),
    (0, "seed", 1),
    (0, "guidance_scale", 7),
    (0, "dtype", "float16"),
    (0, "timesteps", [1, 0.5]),
    (0, "warmup_requests", 1),
    (0, "checkpoint_revision", "different"),
    (0, "decoder", "lightweight"),
    (1, "gpu", "another GPU"),
    (1, "input_sha256", "different"),
    (1, "prompt_sha256", "different"),
    (1, "actions_sha256", "different"),
    (1, "domain_ids", [2] * 95),
    (2, "history_frames", 6),
    (3, "vae_sha256", "different"),
    (0, "trials_s", [1.0] * 4),
    (0, "trials_s", [float("nan")] * 5),
    (0, "trials_s", [0] * 5),
    (0, "startup_s", float("nan")),
    (0, "startup_s", float("inf")),
    (0, "startup_s", 0),
    (0, "startup_s", -1),
    (0, "startup_s", "12"),
    (0, "status", "failed"),
    (0, "backend", "vllm"),
    (0, "video", {}),
):
    changed = deepcopy(rows)
    changed[index][key] = value
    try:
        report.validate(changed)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Must reject mismatched {key}")
for index, key in (
    (0, "gpu"),
    (0, "startup_s"),
    (0, "input_sha256"),
    (0, "prompt_sha256"),
    (0, "actions_sha256"),
    (0, "domain_ids"),
    (2, "vae_sha256"),
):
    changed = deepcopy(rows)
    changed[index].pop(key)
    try:
        report.validate(changed)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Must reject missing {key}")
rows[0]["trials"] = [{"total_s": 2.0}] * 5
rows[0].pop("trials_s")
assert report.format_results(rows) == expected
print("PASS: matching settings, chunk counts, timings and video metadata")

readme = Path(__file__).with_name("README.md").read_text()
for latent_shape in LATENTS.values():
    assert " × ".join(map(str, latent_shape)) in readme
site_spec = importlib.util.spec_from_file_location(
    "site", Path(__file__).with_name("results-site.py")
)
assert site_spec is not None and site_spec.loader is not None
site = importlib.util.module_from_spec(site_spec)
site_spec.loader.exec_module(site)
assert "<script>" not in site.table("| <script> | FPS |\n| --- | --- |\n| GPU | 1 |")
assert "&lt;script&gt;" in site.table("| <script> | FPS |")
assert "PRIVATE_HOST" not in site.table(expected)
assert "48 × 25 × 22 × 40" in site.table(readme)
print("PASS: documented latent shapes and escaped website tables")

assert "<video " in site.table(
    "| [Watch](results/three-seconds/auto/vllm/output.mp4) |"
)
assert "<video " not in site.table("| [Watch](../../private.mp4) |")
assert "<video " not in site.table("| [Watch](javascript:alert(1)) |")
assert 'src="videos/auto-vllm.mp4"' in site.table(
    "| [Watch](results/three-seconds/auto/vllm/output.mp4) |"
)
print("PASS: allowlisted video embeds")

with tempfile.TemporaryDirectory() as temporary:
    site.HERE = Path(temporary)
    (site.HERE / "README.md").write_text(readme)
    (site.HERE / "RESULTS.md").write_text(expected)
    for source in site.VIDEOS:
        video_path = site.HERE / source
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"test MP4")
    page = site.build()
    assert page.count("<video ") == 4
    assert "Startup (s)" in page and page.count("<td>12.35</td>") == 4
    for target in site.VIDEOS.values():
        assert (site.HERE / "website" / target).read_bytes() == b"test MP4"
    assert "PRIVATE_HOST" not in page and "PRIVATE_USER" not in page
print("PASS: standalone website exports four video artifacts")

if importlib.util.find_spec("numpy") is not None:
    import numpy as np

    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        video = save_video(np.zeros((97, 16, 16, 3), dtype=np.uint8), Path(temporary))
        assert video["frames"] == 97 and video["fps"] == 30
        assert video["width"] == 16 and video["height"] == 16
        assert (HERE / video["path"]).is_file()
    print("PASS: MP4 encoding, frame count, FPS and full decode")
