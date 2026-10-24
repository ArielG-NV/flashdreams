# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match the vLLM Auto first-window workload using native FlashDreams inference."""

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import (
    CHUNKS,
    HISTORY,
    LATENTS,
    RESULTS,
    REVISIONS,
    TIMESTEPS,
    TRIALS,
    WARMUP,
    save_video,
)
from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout
from cosmos3_nano_sim_auto.inputs import MultiviewSample, prompt_token_ids
from cosmos_interactive.checkpoint_load import load_exported_checkpoint
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
    wan22_ti2v_5b_vae_state_dict_transform,
)
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
REVISION = REVISIONS["auto"]


def main():
    source = RESULTS / "auto-inputs"
    with np.load(source / "raw.npz") as archive:
        raw = {key: archive[key] for key in ("initial", "controls")}
    raw_metadata = json.loads((source / "raw.json").read_text())
    height, width = 480, 832
    out = RESULTS / "auto/flashdreams"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text('{"status": "failed"}\n')
    started = time.perf_counter()
    network = load_exported_checkpoint(
        f"https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/resolve/{REVISION}/model.safetensors.index.json"
    ).eval()
    assert next(network.parameters()).dtype == torch.bfloat16
    snapshot = Path(
        snapshot_download(
            "nvidia/Cosmos3-Nano-Sim-Auto",
            revision=REVISION,
            allow_patterns=["vae/*"],
        )
    )
    vae_path = snapshot / "vae/diffusion_pytorch_model.safetensors"
    decoder = (
        Wan22TI2V5BVAEDecoderConfig(
            checkpoint_path=str(vae_path),
            state_dict_transform=wan22_ti2v_5b_vae_state_dict_transform,
            dtype=torch.bfloat16,
            use_compile=False,
            use_cuda_graph=False,
        )
        .setup()
        .to("cuda")
        .eval()
    )
    encoder = (
        Wan22TI2V5BVAEEncoderConfig(
            checkpoint_path=str(vae_path),
            state_dict_transform=wan22_ti2v_5b_vae_state_dict_transform,
            dtype=torch.bfloat16,
            use_compile=False,
            use_cuda_graph=False,
        )
        .setup()
        .to("cuda")
        .eval()
    )
    sample = MultiviewSample(
        name="three-second-benchmark",
        model_mode="image2video",
        prompt="",
        resolution="480",
        cameras=(raw_metadata["camera"],),
        control_paths=("prepared",),
        image_paths=("prepared",),
        view_captions=(raw_metadata["caption"],),
    )
    history_frames = HISTORY["auto"] - sample.condition_frames
    report: dict[str, Any] = {
        "status": "failed",
        "backend": "flashdreams",
        "frames": 97,
        "work_frames": 97,
        "generated_frames": 96,
        "ar_chunks": 8,
        "latent_frames_per_chunk": CHUNKS["auto"],
        "history_frames": history_frames,
        "height": height,
        "width": width,
        "steps": 4,
        "latent_shape": LATENTS["auto"],
        "history_capacity": sample.condition_frames + history_frames,
        "timesteps": TIMESTEPS,
        "dtype": "bfloat16",
        "guidance_scale": 1.0,
        "seed": 0,
        "fps": 30,
        "revision": REVISION,
        "input_sha256": hashlib.sha256((source / "raw.npz").read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            json.dumps(raw_metadata, sort_keys=True).encode()
        ).hexdigest(),
        "decoder": "Wan 2.2 full VAE",
        "vae_sha256": hashlib.sha256(vae_path.read_bytes()).hexdigest(),
        "startup_s": time.perf_counter() - started,
        "warmup_requests": WARMUP,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "timing_scope": "raw CPU RGB/control, encoding/tokenization, fresh prefill, eight four-step chunks, full Wan decode, CPU uint8; excludes weights",
        "trials": [],
    }
    geometry = sample.geometry(
        pixel_frames=97, latent_patch_size=2, fps=30, frames_per_chunk=CHUNKS["auto"]
    )

    def tokens(latent):
        _, channels, frames, height, width = latent.shape
        return (
            latent.reshape(1, channels, frames, height // 2, 2, width // 2, 2)
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(-1, 192)
        )

    try:
        with torch.inference_mode():
            for index in range(WARMUP + TRIALS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.cuda.nvtx.range(f"auto.fd.request.{index}"):

                    def encode(array):
                        cache = encoder.initialize_autoregressive_cache()
                        pieces = []
                        start = 0
                        while start < len(array):
                            end = min(len(array), 13 if start == 0 else start + 12)
                            pixels = (
                                torch.from_numpy(array[start:end])
                                .to("cuda", torch.float32)
                                .permute(0, 3, 1, 2)
                            )
                            pieces.append(
                                encoder((pixels / 127.5 - 1).unsqueeze(0), cache=cache)
                            )
                            start = end
                        return torch.cat(pieces, dim=1).transpose(1, 2)

                    prompt = prompt_token_ids(
                        sample, pixel_frames=97, fps=30, per_view_captions=True
                    )
                    request_inputs = {
                        "controls": encode(raw["controls"]),
                        "initial": encode(raw["initial"][None]),
                        "text_ids": prompt.text_ids,
                    }
                    view_text_tokens = prompt.view_text_tokens
                    rollout = ChunkRollout(
                        network,
                        geometry=geometry,
                        control_tokens=tokens(
                            request_inputs["controls"].to("cuda", torch.bfloat16)
                        ),
                        condition_tokens=tokens(
                            request_inputs["initial"].to("cuda", torch.bfloat16)
                        ),
                        text_ids=request_inputs["text_ids"].to("cuda"),
                        view_text_tokens=view_text_tokens,
                        history_frames=history_frames,
                        seed=0,
                        fps=30,
                        t_list=TIMESTEPS,
                        flex_attention=True,
                        accelerated_rms_norm=True,
                        maskless_attention=True,
                        attention_scope="decomposed",
                        decomposed_temporal_window_seconds=0.4,
                    )
                    torch.cuda.synchronize()
                    prefilled = time.perf_counter()
                    traces = []
                    while rollout.next_range is not None:
                        trace, _ = rollout.step()
                        assert trace.denoising_passes == 4
                        traces.append(trace)
                    assert len(traces) == report["ar_chunks"]
                    assert sum(trace.clean_pass for trace in traces) == len(traces) - 1
                    latent = rollout.result().latent
                    report["latent_shape"] = list(latent.shape[1:])
                    assert report["latent_shape"] == LATENTS["auto"]
                    torch.cuda.synchronize()
                    generated = time.perf_counter()
                    decode_cache = decoder.initialize_autoregressive_cache()
                    pieces = [
                        decoder(latent[:, :, :4].transpose(1, 2), cache=decode_cache)
                    ]
                    pieces.extend(
                        decoder(
                            latent[:, :, start : start + 3].transpose(1, 2),
                            cache=decode_cache,
                        )
                        for start in range(4, latent.shape[2], 3)
                    )
                    video = torch.cat(pieces, dim=1)
                    pixels = (
                        ((video[0].float().permute(0, 2, 3, 1) + 1) * 127.5)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                assert pixels.shape == (report["work_frames"], height, width, 3), (
                    pixels.shape
                )
                if index >= WARMUP:
                    report["trials"].append(
                        {
                            "total_s": end - start,
                            "prefill_s": prefilled - start,
                            "denoise_s": generated - prefilled,
                            "decode_cpu_s": end - generated,
                        }
                    )
                print("AUTO_FD_REQUEST", index, end - start, flush=True)
                del rollout, latent, video
        times = [row["total_s"] for row in report["trials"]]
        report.update(
            median_request_s=statistics.median(times),
            p90_request_s=sorted(times)[int(np.ceil(0.9 * len(times))) - 1],
            generated_fps=report["generated_frames"] * len(times) / sum(times),
        )
        report["video"] = save_video(pixels, out)
        report["status"] = "passed"
    finally:
        (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
