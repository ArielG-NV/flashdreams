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

"""FlashDreams-native MIRA diffusion transformer and temporal KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import nvtx
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from flashdreams.core.attention import BlockKVCache
from flashdreams.infra.config import InstantiateConfig

from mira_integration.action import MiraActionEncoder
from mira_integration.modules import (
    AdaptiveLayerNorm,
    MiraSelfAttention,
    SwiGLU,
    spatial_rope,
    temporal_rope,
)


@nvtx.annotate("mira.network.timestep_embedding")
def timestep_embedding(timesteps: Tensor, dimension: int = 256) -> Tensor:
    """Build MIRA's sinusoidal diffusion-time embedding."""
    half = dimension // 2
    exponent = (
        -math.log(10_000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    phase = timesteps.float()[:, None] * torch.exp(exponent)[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class MiraDiffusionTimeEmbedding(nn.Module):
    """Lift scalar flow time into the transformer hidden width."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    @nvtx.annotate("MiraDiffusionTimeEmbedding.forward")
    def forward(self, tau: Tensor) -> Tensor:
        """Encode ``[B,T,1,1,1]`` flow times into ``[B,T,1,1,D]``."""
        batch, frames = tau.shape[:2]
        flat = 1000.0 * tau.reshape(batch * frames)
        dtype = next(self.mlp.parameters()).dtype
        encoded = self.mlp(timestep_embedding(flat).to(dtype=dtype))
        return encoded.reshape(batch, frames, 1, 1, -1)


class MiraSTBlock(nn.Module):
    """Spatial attention, optional temporal attention, and adaptive SwiGLU."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        time_attention: bool,
        attention_gating: bool,
        ada_attention: bool,
        backend: Literal["math", "efficient", "cudnn", "flash"],
    ) -> None:
        super().__init__()
        self.time_attention = time_attention
        self.ada_attention = ada_attention
        self.space_attn_ln = (
            AdaptiveLayerNorm(dim) if ada_attention else nn.LayerNorm(dim)
        )
        self.space_attn = MiraSelfAttention(
            dim, num_heads, num_kv_heads, gating=attention_gating, backend=backend
        )
        self.time_attn_ln: nn.Module | None = None
        self.time_attn: MiraSelfAttention | None = None
        if time_attention:
            self.time_attn_ln = (
                AdaptiveLayerNorm(dim) if ada_attention else nn.LayerNorm(dim)
            )
            self.time_attn = MiraSelfAttention(
                dim, num_heads, num_kv_heads, gating=attention_gating, backend=backend
            )
        self.mlp_ln = AdaptiveLayerNorm(dim)
        self.mlp = SwiGLU(dim)

    @nvtx.annotate("MiraSTBlock.forward")
    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        *,
        spatial_frequencies: tuple[Tensor, Tensor],
        temporal_frequencies: tuple[Tensor, Tensor],
        kv_cache: BlockKVCache | None = None,
        return_kv: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Apply one factorized spatiotemporal transformer block."""
        batch, frames, height, width, channels = x.shape
        with nvtx.annotate("MiraSTBlock.spatial_attention"):
            spatial_x = rearrange(x, "b t h w c -> (b t) (h w) c")
            if self.ada_attention:
                spatial_cond = rearrange(condition, "b t h w c -> (b t) (h w) c")
                spatial_norm = self.space_attn_ln(spatial_x, spatial_cond)
            else:
                spatial_norm = self.space_attn_ln(spatial_x)
            spatial_x = spatial_x + self.space_attn(
                spatial_norm, rotary=spatial_frequencies
            )

        cached: tuple[Tensor, Tensor] | None = None
        if self.time_attn is not None:
            with nvtx.annotate("MiraSTBlock.temporal_attention"):
                assert self.time_attn_ln is not None
                temporal_x = rearrange(
                    spatial_x,
                    "(b t) (h w) c -> (b h w) t c",
                    b=batch,
                    t=frames,
                    h=height,
                    w=width,
                )
                if self.ada_attention:
                    temporal_cond = rearrange(condition, "b t h w c -> (b h w) t c")
                    temporal_norm = self.time_attn_ln(temporal_x, temporal_cond)
                else:
                    temporal_norm = self.time_attn_ln(temporal_x)
                attended = self.time_attn(
                    temporal_norm,
                    rotary=temporal_frequencies,
                    causal=True,
                    kv_cache=kv_cache,
                    return_kv=return_kv,
                )
                if return_kv:
                    temporal_out, cached = attended
                else:
                    temporal_out = attended
                temporal_x = temporal_x + temporal_out
                x = rearrange(
                    temporal_x,
                    "(b h w) t c -> b t h w c",
                    b=batch,
                    h=height,
                    w=width,
                )
        else:
            x = rearrange(
                spatial_x,
                "(b t) (h w) c -> b t h w c",
                b=batch,
                t=frames,
                h=height,
                w=width,
            )
        with nvtx.annotate("MiraSTBlock.mlp"):
            return x + self.mlp(self.mlp_ln(x, condition)), cached


@dataclass(kw_only=True)
class MiraDiTConfig(InstantiateConfig):
    """Config for the checkpoint-compatible MIRA network."""

    _target: type["MiraDiT"] = field(default_factory=lambda: MiraDiT)

    latent_dim: int = 32
    """Codec latent channel count."""

    hidden_dim: int = 2048
    """Transformer residual width."""

    num_action_keys: int = 9
    """Number of checkpoint keyboard-action fields resolved from the manifest."""

    num_heads: int = 16
    """Query head count."""

    num_kv_heads: int = 4
    """Grouped-query key/value head count."""

    num_layers: int = 16
    """Transformer block count."""

    time_attention_every: int = 4
    """Interval between temporal-attention blocks; the final block is always temporal."""

    latent_height: int = 9
    """Manifest-resolved latent grid height."""

    latent_width: int = 16
    """Manifest-resolved latent grid width."""

    n_players: int = 1
    """Number of vertically tiled player views in the joint world state."""

    attention_gating: bool = True
    """Enable checkpoint-trained sigmoid attention gates."""

    ada_attention: bool = True
    """Condition spatial and temporal LayerNorms with action/time embeddings."""

    attention_backend: Literal["math", "efficient", "cudnn", "flash"] = "cudnn"
    """FlashDreams native-attention backend."""


@dataclass(kw_only=True)
class MiraDiTCache:
    """Per-rollout raw temporal keys and values for MIRA's temporal blocks."""

    temporal: list[BlockKVCache | None]
    """One temporal cache per transformer layer; ``None`` for spatial-only layers."""

    context_length: int
    """Number of primed latent frames preceding AR index zero."""

    @nvtx.annotate("MiraDiTCache.before_update")
    def before_update(self, autoregressive_index: int) -> None:
        """Prepare temporal caches for one generated latent frame."""
        chunk_index = self.context_length + autoregressive_index
        for cache in self.temporal:
            if cache is not None:
                cache.before_update(chunk_index)

    @nvtx.annotate("MiraDiTCache.after_update")
    def after_update(self, autoregressive_index: int) -> None:
        """Commit temporal cache bookkeeping for one generated latent frame."""
        chunk_index = self.context_length + autoregressive_index
        for cache in self.temporal:
            if cache is not None:
                cache.after_update(chunk_index)


class MiraDiffusionTransformer(nn.Module):
    """Action-conditioned MIRA flow transformer over channel-last codec latents."""

    def __init__(self, config: MiraDiTConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.latent_tokens_proj = nn.Linear(config.latent_dim, dim)
        self.past_proj = nn.Linear(config.latent_dim, dim)
        self.diffusion_time_embedding = MiraDiffusionTimeEmbedding(dim)
        self.transformer = nn.ModuleList(
            [
                MiraSTBlock(
                    dim,
                    config.num_heads,
                    config.num_kv_heads,
                    time_attention=(index % config.time_attention_every == 0)
                    or index == config.num_layers - 1,
                    attention_gating=config.attention_gating,
                    ada_attention=config.ada_attention,
                    backend=config.attention_backend,
                )
                for index in range(config.num_layers)
            ]
        )
        self.head = nn.Linear(dim, config.latent_dim)

    @nvtx.annotate("MiraDiffusionTransformer.forward")
    def forward(
        self,
        latent: Tensor,
        actions: Tensor,
        tau: Tensor,
        *,
        clean_past: Tensor,
        cache: MiraDiTCache | None = None,
        return_kv: bool = False,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Predict flow and optionally return raw temporal keys/values for priming."""
        _, frames, height, width, _ = latent.shape
        with nvtx.annotate("MiraDiffusionTransformer.input_projection"):
            sequence = self.latent_tokens_proj(latent) + self.past_proj(clean_past)
            condition = actions[:, :, None, None].expand(-1, -1, height, width, -1)
            condition = condition + self.diffusion_time_embedding(tau).expand(
                -1, -1, height, width, -1
            )
        with nvtx.annotate("MiraDiffusionTransformer.rope"):
            spatial_frequencies = spatial_rope(
                height,
                width,
                self.config.hidden_dim // self.config.num_heads,
                latent.device,
            )
            if cache is None:
                temporal_length = frames
            else:
                temporal_length = next(
                    item.size for item in cache.temporal if item is not None
                )
            temporal_frequencies = temporal_rope(
                temporal_length,
                self.config.hidden_dim // self.config.num_heads,
                latent.device,
            )

        returned: list[tuple[Tensor, Tensor] | None] = []
        with nvtx.annotate("MiraDiffusionTransformer.blocks"):
            for index, block in enumerate(self.transformer):
                sequence, raw_kv = block(
                    sequence,
                    condition,
                    spatial_frequencies=spatial_frequencies,
                    temporal_frequencies=temporal_frequencies,
                    kv_cache=cache.temporal[index] if cache is not None else None,
                    return_kv=return_kv,
                )
                if return_kv:
                    returned.append(raw_kv)
        with nvtx.annotate("MiraDiffusionTransformer.output_projection"):
            output = self.head(sequence)
            return (output, returned) if return_kv else output


class MiraDiT(nn.Module):
    """Checkpoint root containing the DiT, learned action encoder, and BOS latent."""

    def __init__(self, config: MiraDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.world_model = MiraDiffusionTransformer(config)
        self.action_encoder = MiraActionEncoder(
            num_keys=config.num_action_keys,
            dim=config.hidden_dim,
            per_player_dropout=config.n_players > 1,
        )
        if config.n_players > 1:
            self.player_embedding = nn.Parameter(
                torch.randn(config.n_players, config.hidden_dim) * 0.02
            )
            self.player_action_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        self.bos = nn.Parameter(
            torch.empty(config.latent_height, config.latent_width, config.latent_dim)
        )
        nn.init.normal_(self.bos, std=0.02)

    @nvtx.annotate("MiraDiT.encode_actions")
    def encode_actions(self, rows: Tensor) -> Tensor:
        """Encode raw action rows and return the final latent-frame token."""
        actions = self.action_encoder(rows)
        if self.config.n_players > 1:
            actions = rearrange(
                actions,
                "(b p) t d -> b p t d",
                p=self.config.n_players,
            )
            actions = actions + self.player_embedding[None, :, None, :]
            actions = self.player_action_projection(actions).mean(dim=1)
        return actions[:, -1:]

    @torch.no_grad()
    @nvtx.annotate("MiraDiT.initialize_cache")
    def initialize_cache(
        self, context_latents: Tensor, context_action_rows: Tensor
    ) -> MiraDiTCache:
        """Prime temporal caches from normalized context latents and action rows."""
        latent = rearrange(context_latents, "b c t h w -> b t h w c")
        batch, frames, height, width, channels = latent.shape
        assert (height, width, channels) == (
            self.config.latent_height,
            self.config.latent_width,
            self.config.latent_dim,
        )
        with nvtx.annotate("MiraDiT.initialize_cache.encode_actions"):
            actions = self.action_encoder(context_action_rows)
            if self.config.n_players > 1:
                actions = rearrange(
                    actions,
                    "(b p) t d -> b p t d",
                    p=self.config.n_players,
                )
                actions = actions + self.player_embedding[None, :, None, :]
                actions = self.player_action_projection(actions).mean(dim=1)
        assert actions.shape[1] == frames, (
            f"context action tokens {actions.shape[1]} != context latents {frames}"
        )
        with nvtx.annotate("MiraDiT.initialize_cache.prime_world_model"):
            clean_past = torch.cat(
                (self.bos[None, None].expand(batch, 1, -1, -1, -1), latent[:, :-1]),
                dim=1,
            )
            tau = torch.ones(
                batch, frames, 1, 1, 1, device=latent.device, dtype=latent.dtype
            )
            _, raw_caches = self.world_model(
                latent,
                actions,
                tau,
                clean_past=clean_past,
                return_kv=True,
            )
        temporal: list[BlockKVCache | None] = []
        with nvtx.annotate("MiraDiT.initialize_cache.materialize_kv_cache"):
            for raw in raw_caches:
                if raw is None:
                    temporal.append(None)
                    continue
                raw_k, raw_v = raw
                cache = BlockKVCache(
                    k_shape=(
                        raw_k.shape[0],
                        frames + 1,
                        raw_k.shape[2],
                        raw_k.shape[3],
                    ),
                    v_shape=(
                        raw_v.shape[0],
                        frames + 1,
                        raw_v.shape[2],
                        raw_v.shape[3],
                    ),
                    seq_dim=1,
                    chunk_size=1,
                    window_size=frames + 1,
                    device=raw_k.device,
                    dtype=raw_k.dtype,
                )
                for index in range(frames):
                    cache.before_update(index)
                    cache.update(
                        raw_k[:, index : index + 1], raw_v[:, index : index + 1]
                    )
                    cache.after_update(index)
                temporal.append(cache)
        return MiraDiTCache(temporal=temporal, context_length=frames)

    @nvtx.annotate("MiraDiT.forward")
    def forward(
        self,
        noisy_tokens: Tensor,
        *,
        timesteps: Tensor,
        cache: MiraDiTCache,
        action_embedding: Tensor,
        clean_past: Tensor,
    ) -> Tensor:
        """Predict current-frame flow from flattened latent tokens."""
        batch = noisy_tokens.shape[0]
        with nvtx.annotate("MiraDiT.forward.reshape_inputs"):
            latent = noisy_tokens.reshape(
                batch,
                1,
                self.config.latent_height,
                self.config.latent_width,
                self.config.latent_dim,
            )
            past = clean_past.reshape_as(latent)
            tau = timesteps.reshape(1, 1, 1, 1, 1).expand(batch, 1, 1, 1, 1)
        with nvtx.annotate("MiraDiT.forward.world_model"):
            output = self.world_model(
                latent,
                action_embedding,
                tau,
                clean_past=past,
                cache=cache,
            )
        with nvtx.annotate("MiraDiT.forward.reshape_output"):
            assert isinstance(output, Tensor)
            return output.reshape(batch, -1, self.config.latent_dim)
