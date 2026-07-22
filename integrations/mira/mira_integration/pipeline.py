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

"""FlashDreams-native MIRA pipeline over the published checkpoint bundle."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import nvtx
import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from mira_integration.decoder import (
    MiraDecoderCache,
    MiraDecoderConfig,
    MiraVideoDecoder,
)
from mira_integration.encoder import (
    MiraBootstrapEncoderConfig,
    MiraControlEncoder,
    MiraControlEncoderCache,
    MiraControlEncoderConfig,
)
from mira_integration.scheduler import MiraFlowScheduler
from mira_integration.transformer import MiraTransformer, MiraTransformerCache
from mira_integration.weights import (
    find_world_checkpoint,
    load_native_weights,
    resolve_bundle,
)

DEFAULT_MODEL_REPO = "alakazamworld/mira-mini"
"""Published MIRA Mini 1B checkpoint bundle used by the demo runner."""


@dataclass(kw_only=True)
class MiraPipelineConfig(StreamInferencePipelineConfig):
    """Config for the fully native MIRA FlashDreams pipeline."""

    _target: type["MiraPipeline"] = field(default_factory=lambda: MiraPipeline)

    encoder: MiraControlEncoderConfig
    """Per-step keyboard encoder."""

    decoder: MiraDecoderConfig
    """Checkpoint-compatible causal video decoder."""

    bootstrap_encoder: MiraBootstrapEncoderConfig = field(
        default_factory=MiraBootstrapEncoderConfig
    )
    """Frozen DINOv3 RAE encoder used once for bootstrap frames."""

    model_repo: str = DEFAULT_MODEL_REPO
    """Hugging Face repository containing the published checkpoint bundle."""

    n_players: int = 1
    """Number of synchronized player views in the checkpoint."""


MiraCache = StreamInferencePipelineCache[
    MiraControlEncoderCache, MiraTransformerCache, MiraDecoderCache
]


class MiraPipeline(
    StreamInferencePipeline[
        MiraControlEncoderCache, MiraTransformerCache, MiraDecoderCache
    ]
):
    """Run MIRA entirely through FlashDreams model and streaming primitives."""

    config: MiraPipelineConfig
    encoder: MiraControlEncoder
    decoder: MiraVideoDecoder

    def __init__(self, config: MiraPipelineConfig) -> None:
        super().__init__(config)
        self.config = config
        assert isinstance(self.encoder, MiraControlEncoder)
        assert isinstance(self.decoder, MiraVideoDecoder)
        assert isinstance(self.diffusion_model.transformer, MiraTransformer)
        self.bootstrap_encoder = config.bootstrap_encoder.setup()
        self._weights_loaded = False
        self._bundle: Path | None = None

    @property
    def transformer(self) -> MiraTransformer:
        """Return the native MIRA transformer with a narrowed type."""
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, MiraTransformer)
        return transformer

    @nvtx.annotate("MiraPipeline._load_weights")
    def _load_weights(self) -> None:
        """Resolve and restore the checkpoint into all native components once."""
        if self._weights_loaded:
            return
        bundle = resolve_bundle(self.config.model_repo, None)
        checkpoint = find_world_checkpoint(bundle)
        load_native_weights(
            checkpoint,
            transformer=self.transformer,
            bootstrap_encoder=self.bootstrap_encoder,
            decoder=self.decoder,
        )
        self.transformer.finish_loading()
        self._bundle = bundle
        self._weights_loaded = True

    @nvtx.annotate("MiraPipeline._context_file")
    def _context_file(self) -> Path:
        """Return the configured bootstrap context file after bundle resolution."""
        assert self._bundle is not None
        path = self._bundle / "context" / "default.npz"
        path = path.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MIRA bootstrap context does not exist: {path}")
        return path

    @nvtx.annotate("MiraPipeline.initialize_cache")
    def initialize_cache(
        self, *, n_diffusion_steps: int = 2, context: str = "default"
    ) -> MiraCache:
        """Load weights, encode bootstrap frames, and prime every streaming cache."""
        if context != "default":
            raise ValueError("MIRA checkpoint bundles provide the default context only")
        self._load_weights()
        with nvtx.annotate("MiraPipeline.initialize_cache.load_context"):
            data = np.load(self._context_file(), allow_pickle=False)
            frames = torch.from_numpy(data["frames"][:, -40:]).to(
                device=self.device, dtype=torch.uint8
            )
            actions = torch.from_numpy(data["actions"][:, -40:]).to(
                device=self.device, dtype=torch.int32
            )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with nvtx.annotate("MiraPipeline.initialize_cache.bootstrap_encode"):
            with torch.no_grad(), autocast:
                latent_window = self.bootstrap_encoder(frames)
            latent_window = latent_window[:, :, -20:].to(
                dtype=self.transformer.config.dtype
            )
            tiled_latent_window = (
                rearrange(
                    latent_window,
                    "(b p) c t h w -> b c t (p h) w",
                    p=self.config.n_players,
                )
                if self.config.n_players > 1
                else latent_window
            )

        scheduler = self.diffusion_model.scheduler
        assert isinstance(scheduler, MiraFlowScheduler)
        scheduler.set_num_inference_steps(n_diffusion_steps)
        with nvtx.annotate("MiraPipeline.initialize_cache.streaming_caches"):
            return StreamInferencePipelineCache(
                encoder_cache=self.encoder.initialize_autoregressive_cache(
                    previous_row=actions[:, -1:]
                ),
                transformer_cache=self.transformer.initialize_autoregressive_cache(
                    context_latents=tiled_latent_window[:, :, 1:],
                    context_action_rows=actions[:, -37:-1],
                ),
                decoder_cache=self.decoder.initialize_autoregressive_cache(
                    context_latents=latent_window
                ),
            )

    def close(self) -> None:
        """Release rollout caches owned by the caller."""

    @torch.no_grad()
    @nvtx.annotate("MiraPipeline.generate")
    def generate(
        self,
        autoregressive_index: int,
        cache: MiraCache,
        input: list[str] | tuple[str, ...] | list[list[str] | None] | None = None,
    ) -> Tensor:
        """Generate two RGB frames for one held-key action step."""
        output = super().generate(
            autoregressive_index,
            cache,
            input=list(input or ()),
        )
        if self.config.n_players == 1:
            assert output.shape[0] == 1
            return output[0]
        assert output.shape[0] == self.config.n_players
        return output


__all__ = [
    "DEFAULT_MODEL_REPO",
    "MiraPipeline",
    "MiraPipelineConfig",
]
