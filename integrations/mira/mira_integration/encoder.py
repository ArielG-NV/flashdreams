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

"""MIRA bootstrap-video and streaming keyboard encoders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from flashdreams.infra.encoder import (
    Encoder,
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)

MIRA_KEYS = ("W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey")
"""Ordered keyboard vocabulary stored in the published checkpoint."""


@dataclass(kw_only=True)
class MiraControlEncoderCache(StreamingEncoderCache):
    """Previous raw action row used to preserve MIRA's two-frame alignment."""

    previous_row: Tensor
    """Multi-hot row ``[B, 1, 9]`` preceding the current control."""


@dataclass(kw_only=True)
class MiraControlEncoderConfig(EncoderConfig):
    """Config for the per-step keyboard encoder."""

    _target: type["MiraControlEncoder"] = field(
        default_factory=lambda: MiraControlEncoder
    )

    valid_keys: tuple[str, ...] = MIRA_KEYS
    """Checkpoint keyboard vocabulary in embedding-table order."""


class MiraControlEncoder(StreamingEncoder[MiraControlEncoderCache]):
    """Convert held-key lists into aligned two-row MIRA action tensors."""

    def __init__(self, config: MiraControlEncoderConfig) -> None:
        super().__init__(config)
        self.mira_config = config
        self._key_index = {name: index for index, name in enumerate(config.valid_keys)}

    def initialize_autoregressive_cache(
        self, *, previous_row: Tensor, **_context: Any
    ) -> MiraControlEncoderCache:
        """Seed the action alignment with the bootstrap context's final row."""
        assert previous_row.shape[-2:] == (1, len(self.mira_config.valid_keys))
        return MiraControlEncoderCache(previous_row=previous_row.to(dtype=torch.int32))

    def forward(
        self,
        input: list[str] | tuple[str, ...] | list[list[str] | None],
        autoregressive_index: int = 0,
        cache: MiraControlEncoderCache | None = None,
    ) -> Tensor:
        """Return ``[previous, current]`` multi-hot rows for one latent step."""
        _ = autoregressive_index
        assert cache is not None, "MIRA controls require an initialized encoder cache"
        current = torch.zeros_like(cache.previous_row)
        if input and (isinstance(input[0], list) or input[0] is None):
            per_player = input
        else:
            flat_input = [key for key in input if isinstance(key, str)]
            per_player = [flat_input] * current.shape[0]
        if len(per_player) != current.shape[0]:
            raise ValueError(
                f"Expected controls for {current.shape[0]} players, got {len(per_player)}"
            )
        for player, keys in enumerate(per_player):
            if keys is None:
                current[player].fill_(-1)
                continue
            for key in keys:
                if key not in self._key_index:
                    raise ValueError(f"Unknown MIRA key: {key!r}")
                current[player, ..., self._key_index[key]] = 1
        rows = torch.cat((cache.previous_row, current), dim=1)
        cache.previous_row = current
        return rows


@dataclass(kw_only=True)
class MiraBootstrapEncoderConfig(EncoderConfig):
    """Config for the frozen DINOv3 RAE bootstrap encoder."""

    _target: type["MiraBootstrapEncoder"] = field(
        default_factory=lambda: MiraBootstrapEncoder
    )

    latent_dim: int = 32
    """Published codec latent width."""

    temporal_stride: int = 2
    """Pixel frames folded into each latent frame."""

    spatial_stride: int = 2
    """DINO patch-grid cells folded into each latent cell."""

    aggregation_layers: tuple[int, ...] = (11, 13, 15, 17, 19, 21, 23)
    """DINO block outputs averaged before adding the final selected output."""

    latent_mean: float = 0.008073142703917577
    """Published codec latent mean."""

    latent_std: float = 0.9276199261988762
    """Published codec latent standard deviation."""


class _DinoBackbone(nn.Module):
    """Checkpoint-owned DINOv3 backbone with a stable state-dict prefix."""

    def __init__(self, layers: tuple[int, ...]) -> None:
        super().__init__()
        logging.getLogger("dinov3").setLevel(logging.WARNING)
        self.layers = layers
        self.dino_model = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vitl16",
            source="github",
            pretrained=False,
            verbose=False,
            trust_repo=True,
        )
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[
                None, :, None, None
            ],
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[
                None, :, None, None
            ],
            persistent=False,
        )

    def forward(self, video: Tensor) -> Tensor:
        """Return the aggregated DINO feature map for ``[B,T,C,H,W]`` video."""
        batch, frames, _, height, width = video.shape
        images = rearrange(video, "b t c h w -> (b t) c h w")
        images = (images - self.get_buffer("mean")) / self.get_buffer("std")
        target = (16 * (height // 16), 16 * (width // 16))
        if images.shape[-2:] != target:
            images = nn.functional.interpolate(
                images, size=target, mode="bilinear", antialias=True
            )
        features = self.dino_model.get_intermediate_layers(
            images, n=self.layers, norm=True, reshape=True
        )
        stacked = torch.stack(tuple(features), dim=0)
        aggregate = stacked.mean(dim=0) + features[-1]
        return rearrange(aggregate, "(b t) c h w -> b t c h w", b=batch, t=frames)


class MiraBootstrapEncoder(Encoder):
    """Encode bootstrap RGB frames into normalized MIRA codec latents."""

    def __init__(self, config: MiraBootstrapEncoderConfig) -> None:
        super().__init__(config)
        self.mira_config = config
        self.rae_projection = nn.Conv3d(
            1024,
            config.latent_dim,
            kernel_size=(
                config.temporal_stride,
                config.spatial_stride,
                config.spatial_stride,
            ),
            stride=(
                config.temporal_stride,
                config.spatial_stride,
                config.spatial_stride,
            ),
        )
        self.rae_dino = _DinoBackbone(config.aggregation_layers)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        """Encode ``[B,T,C,H,W]`` uint8/float RGB into ``[B,C,T,h,w]`` latents."""
        video = (
            input.float().div(255.0) if input.dtype == torch.uint8 else input.float()
        )
        features = self.rae_dino(video)
        latent = self.rae_projection(rearrange(features, "b t c h w -> b c t h w"))
        return (latent - self.mira_config.latent_mean) / self.mira_config.latent_std
