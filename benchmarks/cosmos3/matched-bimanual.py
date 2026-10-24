# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched first-window requests through the existing FlashDreams/vLLM pipelines."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from benchmark import (
    CHUNKS,
    HISTORY,
    RESULTS,
    REVISIONS,
    SEED,
    TRIALS,
    WARMUP,
    git_revision,
    save_video,
)
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent


def flashdreams(image, inputs, actions, report):
    from cosmos3_nano_sim_bimanual.apps.handdreams.adapter import (
        _add_model_arguments,
        _resolve_pipeline_config,
        _setup_inference_pipeline,
    )
    from cosmos3_nano_sim_bimanual.apps.handdreams.scenes import Scene
    from cosmos3_nano_sim_bimanual.encoder import HandActionInput
    from cosmos3_nano_sim_bimanual.tokenizer import tokenize_prompt
    from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
    from flashdreams.recipes.wan.autoencoder.vae import (
        wan22_ti2v_5b_vae_state_dict_transform,
    )

    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    config_args = parser.parse_args(
        [
            "--seed",
            "0",
            "--chunk-size",
            str(CHUNKS["bimanual"]),
            "--decoder",
            "wan",
            "--no-compile-denoise",
            "--no-cuda-graph-denoise",
            "--no-valid-length-cache-attention",
            "--window-frames",
            str(HISTORY["bimanual"] - 1),
            "--denoise-steps",
            "4",
            "--checkpoint",
            f"https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/{REVISIONS['bimanual']}/transformer/diffusion_pytorch_model.safetensors.index.json",
        ]
    )
    pipeline = _setup_inference_pipeline(_resolve_pipeline_config(vars(config_args)))
    assert pipeline.transformer.dtype == torch.bfloat16
    assert list(pipeline.transformer.latent_shape[1:]) == [
        48,
        CHUNKS["bimanual"],
        22,
        40,
    ]
    assert pipeline.transformer.config.window_frames + 1 == HISTORY["bimanual"]
    vae_path = Path(
        hf_hub_download(
            "nvidia/Cosmos3-Nano-Sim-Bimanual",
            "vae/diffusion_pytorch_model.safetensors",
            revision=REVISIONS["bimanual"],
        )
    )
    reference = wan22_ti2v_5b_vae_state_dict_transform(load_file(vae_path))
    for component in (pipeline.scene_encoder.vae.vae, pipeline.decoder.backend.vae):
        assert isinstance(component, torch.nn.Module)
        for name, value in component.named_parameters():
            assert torch.equal(value.cpu(), reference[name].to(value.dtype)), (
                f"VAE weights differ: {name}"
            )
    report.update(
        vae_sha256=hashlib.sha256(vae_path.read_bytes()).hexdigest(),
        latent_shape=[
            pipeline.transformer.latent_shape[1],
            len(actions) // pipeline.temporal_compression_ratio + 1,
            *pipeline.transformer.latent_shape[-2:],
        ],
        history_capacity=pipeline.transformer.config.window_frames + 1,
    )
    scheduler = pipeline.config.diffusion_model.scheduler
    assert isinstance(scheduler, FlowMatchSchedulerConfig)
    assert scheduler.num_inference_steps == 4
    assert scheduler.denoising_timesteps == [1000, 750, 500, 250]
    assert scheduler.shift == 5.0
    counter = 0

    def generate():
        nonlocal counter
        # A unique scene name forces a fresh text/video/decoder cache each trial.
        counter += 1
        rng = pipeline.diffusion_model.rng
        assert rng is not None
        rng.manual_seed(SEED)
        latent = pipeline.patchify_scene_latent(
            pipeline.encode_scene(torch.from_numpy(np.array(image))),
            content_height=image.height,
            content_width=image.width,
        )
        current = Scene(
            name=f"matched-{counter}",
            init_latent=latent,
            init_rgb=np.array(image),
            content_height=image.height,
            content_width=image.width,
            text_ids=tokenize_prompt(inputs["prompt"]).to(pipeline.device),
            fps=inputs["fps"],
            wrist_rotations={},
            gripper_open={},
            wrist_poses={},
        )
        context = {"scene": current}
        cache = pipeline.initialize_cache(
            transformer_context=context, decoder_context=context
        )
        outputs = []
        chunk = 4 * CHUNKS["bimanual"]
        for index, start in enumerate(range(0, len(actions), chunk)):
            action = HandActionInput(
                torch.from_numpy(actions[start : start + chunk]).to(pipeline.device),
                torch.tensor(
                    inputs.get("domain_ids", [inputs["domain_id"]] * len(actions))[
                        start : start + chunk
                    ],
                    device=pipeline.device,
                    dtype=torch.long,
                ),
            )
            outputs.append(pipeline.generate(index, cache, input=action).clone())
            pipeline.finalize(index, cache)
        output = torch.cat(outputs)
        pipeline.release_session(cache)
        pipeline.transformer._prepared_caches.pop(current.name, None)
        assert output.shape == (len(actions), 3, image.height, image.width)
        return output

    return generate, pipeline.close


def main():
    source = RESULTS / "bimanual-inputs"
    out = RESULTS / "bimanual/flashdreams"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text('{"status": "failed"}\n')
    inputs = json.loads((source / "input.json").read_text())
    assert inputs["fps"] == 30
    actions = np.load(source / "actions.npy")[:96]
    assert len(actions) == 96
    image = Image.open(source / "initial.png").convert("RGB").crop((0, 0, 640, 352))

    trials: list[float] = []
    report = {
        "backend": "flashdreams",
        "frames": 97,
        "work_frames": len(actions) + 1,
        "latent_frames_per_chunk": CHUNKS["bimanual"],
        "generated_frames": len(actions),
        "height": image.height,
        "width": image.width,
        "denoise_steps": 4,
        "timesteps": [1.0, 0.9375, 5 / 6, 0.625],
        "dtype": "bfloat16",
        "guidance_scale": 1.0,
        "fps": 30,
        "seed": SEED,
        "decoder": "Wan 2.2 full VAE",
        "output_dtype": "uint8",
        "output_layout": "TCHW CPU",
        "status": "failed",
        "input_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(inputs["prompt"].encode()).hexdigest(),
        "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "domain_id": inputs["domain_id"],
        "domain_ids": inputs.get("domain_ids", [inputs["domain_id"]] * len(actions))[
            : len(actions)
        ],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint_revision": REVISIONS["bimanual"],
        "flashdreams_commit": git_revision(HERE.parents[1]),
        "timing_scope": "image encode, text/cache prefill, generate, full Wan decode, CPU uint8; excludes weights",
        "warmup_requests": WARMUP,
        "trials": trials,
    }
    close = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generate, close = flashdreams(image, inputs, actions, report)
            torch.cuda.synchronize()
            report["startup_s"] = time.perf_counter() - started
            for index in range(TRIALS + WARMUP):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.cuda.nvtx.range("matched.request"):
                    pixels = generate().detach()
                    if pixels.dtype != torch.uint8:
                        assert torch.isfinite(pixels).all()
                        pixels = (
                            ((pixels.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                        )
                    pixels = pixels.cpu()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                assert torch.isfinite(pixels).all()
                if index >= WARMUP:
                    trials.append(elapsed)
                print(index, elapsed, flush=True)
            report["median_request_s"] = statistics.median(trials)
            report["generated_fps"] = len(actions) * TRIALS / sum(trials)
            report["video"] = save_video(
                np.concatenate(
                    [np.asarray(image)[None], pixels.permute(0, 2, 3, 1).numpy()]
                ),
                out,
            )
            report["status"] = "passed"
    finally:
        if close:
            with torch.inference_mode():
                close()
        (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
