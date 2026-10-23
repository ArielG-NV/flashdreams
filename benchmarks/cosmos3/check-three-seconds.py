# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for paired workloads and GPU/FPS-only output."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark import CHUNKS, HISTORY, LATENTS, REVISIONS, SHAPES, TIMESTEPS

spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).with_name("report-three-seconds.py")
)
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
rows: list[dict[str, Any]] = [
    {
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
expected = (
    "GPU: NVIDIA test GPU\n\n| Model | FlashDreams FPS | vLLM FPS |\n"
    "| --- | ---: | ---: |\n| Bimanual | 48.00 | 48.00 |\n| Auto | 48.00 | 48.00 |\n"
)
assert report.format_results(rows) == expected
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
    (0, "status", "failed"),
    (0, "backend", "vllm"),
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
print("PASS: matching settings, actual chunk counts, timings and GPU/FPS-only output")

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
