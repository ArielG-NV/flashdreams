# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched Cosmos3 workload validation and performance/video table."""

import hashlib
import json
import math
import statistics
from typing import Any

from benchmark import (
    CHUNKS,
    GENERATED_FRAMES,
    HERE,
    HISTORY,
    LATENTS,
    RESULTS,
    REVISIONS,
    SHAPES,
    TIMESTEPS,
    TRIALS,
    WARMUP,
    WORK_FRAMES,
)


def trial_times(row: dict[str, Any]) -> list[float]:
    return [
        trial["total_s"] if isinstance(trial, dict) else trial
        for trial in row.get("trials_s", row.get("trials", []))
    ]


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 4:
        raise ValueError("Expected FlashDreams and vLLM results for both models")
    for index, row in enumerate(rows):
        model = "bimanual" if index < 2 else "auto"
        expected = {
            "status": "passed",
            "backend": "flashdreams" if index % 2 == 0 else "vllm",
            "work_frames": WORK_FRAMES,
            "generated_frames": GENERATED_FRAMES,
            "fps": 30,
            "seed": 0,
            "guidance_scale": 1.0,
            "decoder": "Wan 2.2 full VAE",
            "warmup_requests": WARMUP,
            "width": SHAPES[model][0],
            "height": SHAPES[model][1],
            "latent_frames_per_chunk": CHUNKS[model],
            "dtype": "bfloat16",
            "latent_shape": LATENTS[model],
            "history_capacity": HISTORY[model],
            "timesteps": TIMESTEPS,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{model}: invalid or missing {key}")
        if row.get("steps", row.get("denoise_steps")) != 4:
            raise ValueError("Expected four denoising steps")
        if row.get("revision", row.get("checkpoint_revision")) != REVISIONS[model]:
            raise ValueError("Checkpoint revision differs")
        video = row.get("video", {})
        for key, value in {
            "path": f"results/three-seconds/{model}/{row['backend']}/output.mp4",
            "frames": WORK_FRAMES,
            "fps": 30,
            "width": SHAPES[model][0],
            "height": SHAPES[model][1],
        }.items():
            if video.get(key) != value:
                raise ValueError(f"{model}: invalid or missing video {key}")
        if not isinstance(video.get("sha256"), str) or len(video["sha256"]) != 64:
            raise ValueError("Missing video SHA256")
        times = trial_times(row)
        if len(times) != TRIALS or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("Expected five finite, positive request times")
        startup = row.get("startup_s")
        if (
            not isinstance(startup, (int, float))
            or not math.isfinite(startup)
            or startup <= 0
        ):
            raise ValueError("Expected a finite, positive startup time")
        if not row.get("gpu") or row["gpu"] != rows[0]["gpu"]:
            raise ValueError("All runs must use the same GPU model")
    for left, right in ((rows[0], rows[1]), (rows[2], rows[3])):
        for key in ("input_sha256", "prompt_sha256", "vae_sha256"):
            if not left.get(key) or left[key] != right.get(key):
                raise ValueError(f"Backend pair mismatch: {key}")
    for key in ("actions_sha256", "domain_ids"):
        if not rows[0].get(key) or rows[0][key] != rows[1].get(key):
            raise ValueError(f"Bimanual pair mismatch: {key}")
    if len(rows[0]["domain_ids"]) != GENERATED_FRAMES:
        raise ValueError("Expected one domain ID per action row")
    for row in rows[2:]:
        if row.get("history_frames") != 3:
            raise ValueError("Auto must retain one previous three-frame latent chunk")


def format_results(rows: list[dict[str, Any]]) -> str:
    validate(rows)
    lines = [
        f"GPU: {rows[0]['gpu']}",
        "",
        "| Checkpoint | Backend | Latent shape C×T×H×W | Latents/chunk | Denoise steps/chunk | Startup (s) | Generated FPS | Median request (s) | p90 request (s) | Video |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(rows):
        model = "Bimanual" if index < 2 else "Auto"
        times = trial_times(row)
        rate = GENERATED_FRAMES * TRIALS / sum(times)
        latent = " × ".join(map(str, row["latent_shape"]))
        backend = "FlashDreams" if index % 2 == 0 else "vLLM-Omni"
        video = row["video"]["path"]
        lines.append(
            f"| Cosmos3-Nano-Sim-{model} | {backend} | {latent} | "
            f"{row['latent_frames_per_chunk']} | 4 | {row['startup_s']:.2f} | {rate:.2f} | "
            f"{statistics.median(times):.2f} | {sorted(times)[math.ceil(0.9 * len(times)) - 1]:.2f} | "
            f"[Watch]({video}) |"
        )
    lines.extend(
        [
            "",
            "Startup: saved benchmark.json startup_s, covering model/backend initialization before the first warmup request; excludes process launch, input preparation and request warmup. Cache state can affect startup time.",
            "",
            (
                "Videos: 97 frames at 30 FPS, including the initial frame; last measured request, seed 0. "
                "MP4 encoding is excluded from timings. Auto/AV uses one RGB camera without LiDAR. "
                "Runs use eager denoising with CUDA graphs off; these are not the notebooks' optimized serving settings. "
                "Quality equivalence is unverified. Measurements on a shared GPU can vary."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [
        json.loads((RESULTS / model / backend / "benchmark.json").read_text())
        for model in REVISIONS
        for backend in ("flashdreams", "vllm")
    ]
    output = format_results(rows)
    for row in rows:
        video = row["video"]
        if (
            hashlib.sha256((HERE / video["path"]).read_bytes()).hexdigest()
            != video["sha256"]
        ):
            raise ValueError("Saved video hash differs from the measured run")
    (HERE / "RESULTS.md").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(f"Cannot compare benchmark results: {error}") from None
