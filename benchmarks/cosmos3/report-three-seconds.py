# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched Cosmos3 workload validation and GPU/FPS results."""

import json
import math
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
        times = trial_times(row)
        if len(times) != TRIALS or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("Expected five finite, positive request times")
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
        "| Model | FlashDreams FPS | vLLM FPS |",
        "| --- | ---: | ---: |",
    ]
    for offset, model in ((0, "Bimanual"), (2, "Auto")):
        rates = [
            GENERATED_FRAMES * TRIALS / sum(trial_times(row))
            for row in rows[offset : offset + 2]
        ]
        lines.append(f"| {model} | {rates[0]:.2f} | {rates[1]:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [
        json.loads((RESULTS / model / backend / "benchmark.json").read_text())
        for model in REVISIONS
        for backend in ("flashdreams", "vllm")
    ]
    output = format_results(rows)
    (HERE / "RESULTS.md").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(f"Cannot compare benchmark results: {error}") from None
