# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline precompute of alpadreams text + first-frame image embeddings.

Loads ONLY the one-shot encoders (Cosmos-Reason1-7B text encoder and
the Wan VAE first-frame image encoder) from the same builder used by
:mod:`examples.run_alpadreams`, runs them on the demo prompts/first
frames, and saves the resulting embeddings to a ``.pt`` file. The DiT
and per-AR-step encoder/decoder are NOT loaded.

Pair with ``run_alpadreams.py --embeddings_path <path>``: that path
constructs the pipeline with ``text_encoder=None`` /
``image_encoder=None`` (saving ~14 GB of VRAM throughout the rollout)
and hydrates the cache via
:meth:`AlpadreamsPipeline.initialize_cache_from_embeddings`.

Run::

    python examples/precompute_alpadreams_embeddings.py \\
        --n_cameras 1 \\
        --output outputs/alpadreams_sv_embeddings.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import mediapy as media
import torch
from einops import rearrange

# ``examples/`` is not a package, so import the sibling demo by adding
# this directory to sys.path. Lets us share data definitions
# (``_build_data``, S3 paths) with run_alpadreams.py instead of
# duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_alpadreams import (  # noqa: E402
    EXAMPLE_DATA_DIR_LOCAL,
    EXAMPLE_DATA_DIR_S3,
    REPO_ROOT,
    _build_data,
)

from flashdreams.core.io.s3_sync import sync_s3_dir_to_local  # noqa: E402
from flashdreams.recipes.alpadreams.config import (  # noqa: E402
    ALPADREAMS_CONFIG_BUILDERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_cameras", type=int, default=1, help="Number of cameras (1 or 4)."
    )
    parser.add_argument(
        "--overwrite_config_name",
        type=str,
        default=None,
        choices=sorted(ALPADREAMS_CONFIG_BUILDERS.keys()) + [None],  # type: ignore[arg-type]
        help="Optionally override the per-n_cameras default config name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .pt path. Defaults to "
            "outputs/alpadreams_<config_name>_embeddings.pt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.n_cameras in (1, 4), "Only 1 or 4 cameras are supported"

    config_meta, data = _build_data(args.n_cameras)
    config_name = (
        args.overwrite_config_name
        if args.overwrite_config_name is not None
        else config_meta[0]
    )
    camera_names = config_meta[1:]
    output_path = args.output or (
        f"{REPO_ROOT}/outputs/alpadreams_{config_name}_embeddings.pt"
    )

    print(
        f"Precomputing alpadreams embeddings for {args.n_cameras} cameras "
        f"with config: {config_name}"
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    credential_path = str(REPO_ROOT / "credentials/s3_checkpoint.secret")
    assert os.path.exists(credential_path), (
        f"Credential file not found at {credential_path}"
    )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=credential_path,
        cache_dir=EXAMPLE_DATA_DIR_LOCAL,
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing from S3",
    )

    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is not set"

    # Build the same config the inference run will use, but only setup()
    # the two one-shot encoders -- DiT, per-AR-step encoder, and decoder
    # are intentionally NOT loaded here (they're not needed to produce
    # the embeddings, and skipping them keeps precompute lightweight).
    builder = ALPADREAMS_CONFIG_BUILDERS[config_name]
    pipeline_config = builder(cp_size=1, compile_network=False, seed=0)

    assert (
        pipeline_config.text_encoder is not None
        and pipeline_config.image_encoder is not None
    ), (
        "Cannot precompute: the chosen config has text_encoder/image_encoder "
        "set to None. Use a config that keeps both encoders configured."
    )

    # Pixel-space resolution required by the first-frame encoder, derived
    # from the same (transformer.height/width, decoder.SPATIAL_COMPRESSION)
    # logic as run_alpadreams.py -- read off the configs without
    # instantiating the transformer or decoder.
    transformer_cfg = pipeline_config.diffusion_model.transformer
    decoder_sp = pipeline_config.decoder._target.SPATIAL_COMPRESSION_RATIO  # type: ignore[union-attr]
    pixel_h = transformer_cfg.height * decoder_sp
    pixel_w = transformer_cfg.width * decoder_sp

    text_encoder = pipeline_config.text_encoder.setup().to(device=device)
    image_encoder = pipeline_config.image_encoder.setup().to(device=device)

    first_frames: list[torch.Tensor] = []
    prompts: list[str] = []
    for entry in data:
        first_frame = media.read_image(entry["first_frame_path"])
        first_frame = cv2.resize(first_frame, (pixel_w, pixel_h))
        first_frame_t = (
            torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
        )
        first_frames.append(rearrange(first_frame_t, "h w c -> 1 c h w"))
        prompts.append(entry["prompt"])

    first_frames_t = torch.stack(first_frames, dim=0).unsqueeze(
        0
    )  # [B=1, V, 1, C, H, W]
    prompts_2d: list[list[str]] = [prompts]  # [B=1, V]

    with torch.no_grad():
        text_embeddings = torch.stack(
            [text_encoder(t) for t in prompts_2d], dim=0
        )  # [B, V, L, D]
        image_embeddings = image_encoder(first_frames_t)  # [B, V, 1, Cl, Hl, Wl]

    payload = {
        "text_embeddings": text_embeddings.cpu(),
        "image_embeddings": image_embeddings.cpu(),
        "view_names": camera_names,
        "metadata": {
            "config_name": config_name,
            "n_cameras": args.n_cameras,
            "prompts": prompts,
            "pixel_h": pixel_h,
            "pixel_w": pixel_w,
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"saved precomputed embeddings to {output_path} "
        f"(text {tuple(text_embeddings.shape)} {text_embeddings.dtype}, "
        f"image {tuple(image_embeddings.shape)} {image_embeddings.dtype})"
    )


if __name__ == "__main__":
    main()
