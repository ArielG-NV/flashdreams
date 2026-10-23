# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-second full rollouts through the released vLLM Cosmos3 Sim pipelines."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from benchmark import (
    CHUNKS,
    HISTORY,
    LATENTS,
    RESULTS,
    REVISIONS,
    SHAPES,
    TIMESTEPS,
    TRIALS,
    WARMUP,
)
from huggingface_hub import snapshot_download
from PIL import Image
from vllm_omni.entrypoints.omni import Omni  # ty: ignore[unresolved-import]
from vllm_omni.inputs.data import (  # ty: ignore[unresolved-import]
    OmniDiffusionSamplingParams,
)

HERE = Path(__file__).resolve().parent


def cpu_pixels(output):
    value = output[0].images
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    while isinstance(value, dict):
        value = value.get("payload", value.get("video", value.get("frames")))
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 5:
            value = value[0]
        if value.shape[0] == 3:
            value = value.permute(1, 2, 3, 0)
        if value.is_floating_point():
            value = ((value.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        value = value.numpy()
    pixels = np.asarray(value)
    while pixels.ndim > 4 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.dtype != np.uint8:
        pixels = (pixels * 255).round().clip(0, 255).astype(np.uint8)
    return pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=REVISIONS)
    parser.add_argument("--init-timeout", type=int, default=1800)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.init_timeout < 1:
        parser.error("init-timeout must be positive")
    kind = args.model
    out = RESULTS / kind / "vllm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text('{"status": "failed"}\n')
    extra: dict[str, Any] = {
        "session_id": "three-second-benchmark",
        "reset": True,
        "close_session": True,
    }
    if kind == "bimanual":
        source = HERE / "results/three-seconds/bimanual-inputs"
        metadata = json.loads((source / "input.json").read_text())
        assert metadata["fps"] == 30
        actions = np.load(source / "actions.npy")[:96]
        image = Image.open(source / "initial.png").convert("RGB").crop((0, 0, 640, 352))
        height, width = 352, 640
        extra.update(
            action_space="normalized_unified_v1",
            action=actions[:, :59].tolist(),
            domain_id=metadata["domain_ids"][:96],
            resolution=480,
        )
        prompt = {
            "prompt": metadata["prompt"],
            "modalities": ["video"],
            "multi_modal_data": {"image": image},
        }
        hashes = {
            "input_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(metadata["prompt"].encode()).hexdigest(),
            "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "domain_id": metadata["domain_id"],
            "domain_ids": metadata["domain_ids"][:96],
        }
    else:
        source = HERE / "results/three-seconds/auto-inputs"
        raw = np.load(source / "raw.npz")
        metadata = json.loads((source / "raw.json").read_text())
        height, width = 480, 832
        control = torch.from_numpy(raw["controls"]).permute(3, 0, 1, 2)
        initial = torch.from_numpy(raw["initial"]).permute(2, 0, 1).unsqueeze(1)
        from vllm_omni.diffusion.models.cosmos3.transfer import (  # ty: ignore[unresolved-import]
            media_to_uint8_cthw,
        )

        for value, frames in ((initial, 1), (control, 97)):
            converted = media_to_uint8_cthw(
                value, height=height, width=width, max_frames=frames
            )
            assert converted.shape == (3, frames, height, width)
            assert torch.equal(converted, value)
        caption = (
            metadata["caption"]
            if isinstance(metadata["caption"], str)
            else json.dumps(metadata["caption"])
        )
        extra.update(
            wsm={},
            multiview={
                "resolution": "480",
                "aspect_ratio": "16,9",
                "num_frames": 97,
                "condition_video_as_image": True,
                "views": [
                    {
                        "camera_key": metadata["camera"],
                        "control": control,
                        "vision": initial,
                        "prompt": caption,
                    }
                ],
            },
        )
        prompt = {
            "prompt": "Prepared residential driving scene.",
            "modalities": ["video"],
        }
        hashes = {
            "input_sha256": hashlib.sha256(
                (source / "raw.npz").read_bytes()
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest(),
        }
    if args.check_inputs:
        print(f"PASS: official {kind} request inputs; {width}x{height}, 97 work frames")
        return
    model_id = "nvidia/Cosmos3-Nano-Sim-" + (
        "Bimanual" if kind == "bimanual" else "Auto"
    )
    snapshot = snapshot_download(
        model_id,
        revision=REVISIONS[kind],
        allow_patterns=[
            "*.json",
            "transformer/*",
            "vae/*",
            "text_tokenizer/*",
            "lidar_vae/*",
            "lidar_tokenizer/*",
        ],
    )
    manifest = json.loads((Path(snapshot) / "transformer/config.json").read_text())[
        f"cosmos3_nano_sim_{kind}"
    ]
    chunk_size = (
        manifest["chunk_size"]
        if kind == "bimanual"
        else manifest["chunk"]["frames_per_chunk"]
    )
    assert chunk_size == CHUNKS[kind], "Checkpoint chunk size changed"
    assert manifest["fixed_step_sampler_config"]["t_list"] == TIMESTEPS
    if kind == "auto":
        assert manifest["ring"] == {"cache_chunks": 2, "sink_chunks": 1}
    else:
        assert manifest["window_frames"] is None and manifest["sink_frames"] == 0
    vae = json.loads((Path(snapshot) / "vae/config.json").read_text())
    latent_frames = (97 - 1) // vae["scale_factor_temporal"] + 1
    latent_shape = [
        vae["z_dim"],
        latent_frames,
        height // vae["scale_factor_spatial"],
        width // vae["scale_factor_spatial"],
    ]
    history_capacity = (
        latent_frames
        if kind == "bimanual"
        else 1
        + (manifest["ring"]["cache_chunks"] - manifest["ring"]["sink_chunks"])
        * chunk_size
    )
    assert latent_shape == LATENTS[kind] and history_capacity == HISTORY[kind]
    official = HERE / "vendor/official-image/app/vllm-omni"
    deployment = yaml.safe_load(
        (official / f"vllm_omni/deploy/cosmos3_nano_sim_{kind}.yaml").read_text()
    )
    assert deployment["dtype"] == "bfloat16"
    assert (width, height) == SHAPES[kind]
    config = deployment["stages"][0].setdefault("model_config", {})
    config.update(guardrails=False, use_cuda_graphs=False)
    config["ar_diffusion_kv_config"] = {
        "gpu_memory_fraction": 0.15,
        "warmup_cudagraph": False,
    }
    deployment_path = out / "deploy.yaml"
    deployment_path.write_text(yaml.safe_dump(deployment))
    report: dict[str, Any] = dict(
        status="failed",
        backend="vllm",
        implementation="official Cosmos3 image pipeline; AR-Diffusion engine; eager",
        model=model_id,
        revision=REVISIONS[kind],
        checkpoint_revision=REVISIONS[kind],
        frames=97,
        work_frames=97,
        generated_frames=96,
        latent_frames_per_chunk=chunk_size,
        height=height,
        width=width,
        steps=4,
        latent_shape=latent_shape,
        history_capacity=history_capacity,
        timesteps=manifest["fixed_step_sampler_config"]["t_list"],
        dtype=deployment["dtype"],
        denoise_steps=4,
        guidance_scale=1.0,
        fps=30,
        seed=0,
        decoder="Wan 2.2 full VAE",
        vae_sha256=hashlib.sha256(
            (Path(snapshot) / "vae/diffusion_pytorch_model.safetensors").read_bytes()
        ).hexdigest(),
        warmup_requests=WARMUP,
        trials_s=[],
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        image_provenance=json.loads(
            (HERE / "results/official-image/provenance.json").read_text()
        ),
        timing_scope="raw CPU inputs, encoding/tokenization, fresh AR rollout, full Wan decode and uint8 CPU pixels; excludes model loading and MP4",
        **hashes,
    )
    if kind == "auto":
        report["history_frames"] = history_capacity - 1
    report["init_timeout_s"] = args.init_timeout
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    model = None
    try:
        started = time.perf_counter()
        model = Omni(
            model=snapshot,
            model_class_name="Cosmos3NanoSim"
            + ("Bimanual" if kind == "bimanual" else "Auto")
            + "Pipeline",
            deploy_config=str(deployment_path),
            enforce_eager=True,
            init_timeout=args.init_timeout,
            stage_init_timeout=args.init_timeout,
        )
        report["startup_s"] = time.perf_counter() - started
        for index in range(WARMUP + TRIALS):
            sampling = OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=97,
                num_inference_steps=4,
                guidance_scale=1.0,
                seed=0,
                fps=30,
                frame_rate=30.0,
                extra_args=extra,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.nvtx.range(f"official.{kind}.{index}"):
                pixels = cpu_pixels(model.generate(prompt, sampling))
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert pixels.shape == (97, height, width, 3), pixels.shape
            if index >= WARMUP:
                report["trials_s"].append(elapsed)
            print("OFFICIAL_REQUEST", kind, index, elapsed, flush=True)
        times = report["trials_s"]
        report.update(
            median_request_s=statistics.median(times),
            generated_fps=96 * len(times) / sum(times),
        )
        report["status"] = "passed"
    finally:
        (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
