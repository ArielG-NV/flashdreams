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

"""FlashDreams streaming decoder for MIRA's checkpoint-owned causal video ViT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import nvtx
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoderCache,
    StreamingVideoDecoder,
)
from mira_integration.modules import LayerScale, MiraSelfAttention, SwiGLU, decoder_rope

DecoderFrequencies = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]


@nvtx.annotate("mira.decoder._spatial_decoder_rope")
def _spatial_decoder_rope(
    height: int, width: int, head_dim: int, theta: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build the codec decoder's axial 2D RoPE frequencies."""
    half = head_dim // 2
    rows = torch.arange(height, device=device).repeat_interleave(width)
    columns = torch.arange(width, device=device).repeat(height)
    cos_h, sin_h = decoder_rope(rows, half, theta)
    cos_w, sin_w = decoder_rope(columns, half, theta)
    return torch.cat((cos_h, cos_w), -1), torch.cat((sin_h, sin_w), -1)


class MiraDecoderBlock(nn.Module):
    """Factorized causal space-time block used by the MIRA codec decoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        backend: Literal["math", "efficient", "cudnn", "flash"],
        causal_temporal_attention_backend: Literal["torch", "triton"],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.space_norm = nn.LayerNorm(dim, eps=eps)
        self.space_attn = MiraSelfAttention(dim, num_heads, num_heads, backend=backend)
        self.space_ls = LayerScale(dim)
        self.time_norm = nn.LayerNorm(dim, eps=eps)
        self.time_attn = MiraSelfAttention(
            dim,
            num_heads,
            num_heads,
            backend=backend,
            causal_attention_backend=causal_temporal_attention_backend,
        )
        self.time_ls = LayerScale(dim)
        self.mlp_norm = nn.LayerNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim)
        self.mlp_ls = LayerScale(dim)

    def forward(
        self,
        x: Tensor,
        spatial_frequencies: tuple[Tensor, Tensor],
        temporal_frequencies: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Apply spatial attention, causal temporal attention, and SwiGLU."""
        batch, frames, height, width, _ = x.shape
        spatial_x = rearrange(x, "b t h w c -> (b t) (h w) c")
        spatial_x = self.space_ls.residual(
            spatial_x,
            self.space_attn(self.space_norm(spatial_x), rotary=spatial_frequencies),
        )
        temporal_x = rearrange(
            spatial_x,
            "(b t) (h w) c -> (b h w) t c",
            b=batch,
            t=frames,
            h=height,
            w=width,
        )
        temporal_x = self.time_ls.residual(
            temporal_x,
            self.time_attn(
                self.time_norm(temporal_x),
                rotary=temporal_frequencies,
                causal=True,
            ),
        )
        x = rearrange(
            temporal_x,
            "(b h w) t c -> b t h w c",
            b=batch,
            h=height,
            w=width,
        )
        return self.mlp_ls.residual(x, self.mlp(self.mlp_norm(x)))


class MiraPatchUnembed(nn.Module):
    """Project ViT tokens into two RGB frames per latent frame."""

    def __init__(self, width: int, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(width, 3 * 2 * patch_size * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Fold projected tokens into ``[B,2T,3,H,W]`` video."""
        projected = self.proj(x)
        patch = self.patch_size
        return rearrange(
            projected,
            "b t h w (c pt ph pw) -> b (t pt) c (h ph) (w pw)",
            c=3,
            pt=2,
            ph=patch,
            pw=patch,
        )


@dataclass(kw_only=True)
class MiraDecoderCache(StreamingDecoderCache):
    """Two-latent causal context carried between decoder calls."""

    latent_history: Tensor
    """Normalized codec latents ``[B,C,2,h,w]`` preceding the current latent."""

    initial_latent_history: Tensor
    """Bootstrap latent history restored before each browser session."""


@dataclass(kw_only=True)
class MiraDecoderConfig(DecoderConfig):
    """Config for the published MIRA causal ViT decoder."""

    _target: type["MiraVideoDecoder"] = field(default_factory=lambda: MiraVideoDecoder)

    latent_dim: int = 32
    """Codec latent width."""

    width: int = 1152
    """Decoder ViT residual width."""

    depth: int = 28
    """Decoder block count."""

    num_heads: int = 16
    """Decoder attention head count."""

    patch_size: int = 16
    """Pixel patch side emitted per decoder token."""

    dtype: torch.dtype = torch.bfloat16
    """Parameter and activation dtype."""

    attention_backend: Literal["math", "efficient", "cudnn", "flash"] = "cudnn"
    """FlashDreams native-attention backend."""

    causal_temporal_attention_backend: Literal["torch", "triton"] = "triton"
    """Backend for decoder no-cache causal temporal attention."""

    compile_core: bool = True
    """Compile the stateless decoder core to fuse elementwise operations."""

    use_cuda_graph: bool = True
    """Replay fixed-shape decode-core calls through a CUDA graph on CUDA devices."""

    cuda_graph_warmup_iters: int = 2
    """Eager decode-core calls before CUDA graph capture."""

    latent_mean: float = 0.008073142703917577
    """Published codec latent mean."""

    latent_std: float = 0.9276199261988762
    """Published codec latent standard deviation."""

    n_players: int = 1
    """Number of player views stacked along the incoming latent height."""


class MiraVideoDecoder(StreamingVideoDecoder[MiraDecoderCache]):
    """Decode MIRA latents with two-frame causal context and emit RGB video."""

    def __init__(self, config: MiraDecoderConfig) -> None:
        super().__init__(config)
        self.mira_config = config
        self.from_latent = nn.ConvTranspose2d(
            config.latent_dim, config.width, kernel_size=2, stride=2
        )
        self.blocks = nn.ModuleList(
            [
                MiraDecoderBlock(
                    config.width,
                    config.num_heads,
                    backend=config.attention_backend,
                    causal_temporal_attention_backend=(
                        config.causal_temporal_attention_backend
                    ),
                )
                for _ in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.width, eps=1e-6)
        self.patch_unembed = MiraPatchUnembed(config.width, config.patch_size)
        self._rope_cache: dict[tuple[int, int, int, int, str], DecoderFrequencies] = {}
        self._decode_core_runner: Callable[
            [Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]], Tensor
        ] = self._decode_core
        self._core_compiled = False
        self._decode_graph = (
            CUDAGraphWrapper(
                self._decode_latent,
                warmup_iters=config.cuda_graph_warmup_iters,
            )
            if config.use_cuda_graph
            else None
        )
        self.to(dtype=config.dtype).eval()

    def finish_loading(self) -> None:
        """Compile the stateless decoder core after checkpoint restoration."""
        if self.mira_config.compile_core and not self._core_compiled:
            self._decode_core_runner = torch.compile(
                self._decode_core,
                mode="max-autotune-no-cudagraphs",
            )
            self._core_compiled = True

    @property
    def spatial_compression_ratio(self) -> int:
        """Return the codec's 32x spatial compression ratio."""
        return 32

    @property
    def temporal_compression_ratio(self) -> int:
        """Return the codec's two-pixel-frames-per-latent ratio."""
        return 2

    @nvtx.annotate("MiraVideoDecoder.get_output_temporal_size")
    def get_output_temporal_size(
        self, autoregressive_index: int, input_temporal_size: int
    ) -> int:
        """Map current latent count to newly emitted pixel frames."""
        _ = autoregressive_index
        return 2 * input_temporal_size

    @nvtx.annotate("MiraVideoDecoder.get_input_temporal_size")
    def get_input_temporal_size(
        self, autoregressive_index: int, output_temporal_size: int
    ) -> int:
        """Map an even requested pixel count to required current latents."""
        _ = autoregressive_index
        assert output_temporal_size % 2 == 0
        return output_temporal_size // 2

    @nvtx.annotate("MiraVideoDecoder.initialize_autoregressive_cache")
    def initialize_autoregressive_cache(
        self, *, context_latents: Tensor, **_context: Any
    ) -> MiraDecoderCache:
        """Seed the decoder with the final two normalized context latents."""
        assert context_latents.shape[2] >= 2
        latent_history = context_latents[:, :, -2:].detach()
        return MiraDecoderCache(
            latent_history=latent_history,
            initial_latent_history=latent_history.clone(),
        )

    @nvtx.annotate("MiraVideoDecoder.restore_autoregressive_cache")
    def restore_autoregressive_cache(self, cache: MiraDecoderCache) -> None:
        """Restore causal decoder history while retaining the captured graph."""
        cache.latent_history = cache.initial_latent_history.clone()

    def _frequencies(
        self,
        *,
        frames: int,
        height: int,
        width: int,
        head_dim: int,
        device: torch.device,
    ) -> DecoderFrequencies:
        key = (frames, height, width, head_dim, str(device))
        cached = self._rope_cache.get(key)
        if cached is None:
            with nvtx.annotate("MiraVideoDecoder.rope"):
                cached = (
                    _spatial_decoder_rope(height, width, head_dim, 100.0, device),
                    decoder_rope(torch.arange(frames, device=device), head_dim, 64.0),
                )
            self._rope_cache[key] = cached
        return cached

    @nvtx.annotate("MiraVideoDecoder._decode_latent")
    def _decode_latent(self, latent: Tensor) -> Tensor:
        height = latent.shape[-2] * self.from_latent.stride[0]
        width = latent.shape[-1] * self.from_latent.stride[1]
        head_dim = self.mira_config.width // self.mira_config.num_heads
        spatial_frequencies, temporal_frequencies = self._frequencies(
            frames=latent.shape[1],
            height=height,
            width=width,
            head_dim=head_dim,
            device=latent.device,
        )
        with nvtx.annotate("MiraVideoDecoder.compiled_core"):
            return self._decode_core_runner(
                latent,
                spatial_frequencies,
                temporal_frequencies,
            )

    def _decode_core(
        self,
        latent: Tensor,
        spatial_frequencies: tuple[Tensor, Tensor],
        temporal_frequencies: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Decode one fixed-shape latent window without mutating streaming state."""
        batch, frames = latent.shape[:2]
        x = rearrange(latent, "b t c h w -> (b t) c h w")
        x = self.from_latent(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=batch, t=frames)
        for block in self.blocks:
            x = block(x, spatial_frequencies, temporal_frequencies)
        video = torch.tanh(self.patch_unembed(self.norm_out(x)))
        return video[:, -2:].mul(0.5).add(0.5)

    @torch.no_grad()
    @nvtx.annotate("MiraVideoDecoder.forward")
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: MiraDecoderCache | None = None,
    ) -> Tensor:
        """Decode one normalized latent and return two new RGB frames."""
        _ = autoregressive_index
        assert cache is not None
        with nvtx.annotate("MiraVideoDecoder.prepare_latents"):
            if self.mira_config.n_players > 1:
                input = rearrange(
                    input,
                    "b c t (p h) w -> (b p) c t h w",
                    p=self.mira_config.n_players,
                )
            normalized = torch.cat((cache.latent_history, input), dim=2)
            cache.latent_history = normalized[:, :, -2:].detach()
            latent = (
                self.mira_config.latent_std * normalized + self.mira_config.latent_mean
            )
            latent = rearrange(latent, "b c t h w -> b t c h w").to(
                dtype=self.mira_config.dtype
            )
        if self._decode_graph is not None and latent.is_cuda:
            return self._decode_graph(latent)
        return self._decode_latent(latent)
