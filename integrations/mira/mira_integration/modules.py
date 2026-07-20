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

"""Checkpoint-compatible MIRA attention, normalization, RoPE, and MLP modules."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.attention import BlockKVCache, NativeAttention


def apply_rotary(x: Tensor, frequencies: tuple[Tensor, Tensor]) -> Tensor:
    """Apply pairwise rotary frequencies to ``[B, S, H, D]`` attention states."""
    cos, sin = (frequency.unsqueeze(-2) for frequency in frequencies)
    paired = x.float().unflatten(-1, (-1, 2))
    real, imag = paired.unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(-2)
    return (x.float() * cos + rotated * sin).to(dtype=x.dtype)


def temporal_rope(
    length: int, head_dim: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build MIRA's seconds-scaled temporal RoPE table."""
    positions = torch.arange(length, dtype=torch.float32, device=device) / 10.0
    bands = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
    inv_freq = torch.exp(bands * (-math.log(64.0) * 2.0 / head_dim))
    phase = positions[:, None] * inv_freq[None, :]
    return phase.cos().repeat_interleave(2, -1), phase.sin().repeat_interleave(2, -1)


def spatial_rope(
    height: int, width: int, head_dim: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build MIRA's axial 2D spatial RoPE table."""
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
    n_freqs = head_dim // 4
    lo, hi = 2.0 * math.pi / 100.0, math.pi
    band = torch.arange(n_freqs, dtype=torch.float32, device=device)
    inv_freq = lo * (hi / lo) ** (band / max(n_freqs - 1, 1))

    def axis(size: int) -> tuple[Tensor, Tensor]:
        phase = (
            torch.arange(size, dtype=torch.float32, device=device)[:, None] * inv_freq
        )
        return phase.cos().repeat_interleave(2, -1), phase.sin().repeat_interleave(
            2, -1
        )

    cos_h, sin_h = axis(height)
    cos_w, sin_w = axis(width)
    cos_h = cos_h[:, None].expand(height, width, -1)
    sin_h = sin_h[:, None].expand(height, width, -1)
    cos_w = cos_w[None].expand(height, width, -1)
    sin_w = sin_w[None].expand(height, width, -1)
    return (
        torch.cat((cos_h, cos_w), -1).reshape(height * width, head_dim),
        torch.cat((sin_h, sin_w), -1).reshape(height * width, head_dim),
    )


def decoder_rope(
    positions: Tensor, head_dim: int, theta: float
) -> tuple[Tensor, Tensor]:
    """Build the codec decoder's standard 1D RoPE table."""
    band = torch.arange(head_dim // 2, dtype=torch.float32, device=positions.device)
    phase = positions.float()[:, None] * theta ** (-2.0 * band[None] / head_dim)
    return phase.cos().repeat_interleave(2, -1), phase.sin().repeat_interleave(2, -1)


class QKLayerNorm(nn.Module):
    """Per-head LayerNorm with MIRA's checkpoint parameter layout."""

    def __init__(self, heads: int, head_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.qk_scale = nn.Parameter(torch.ones(heads, head_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the final channel in fp32 and restore the input dtype."""
        dtype = x.dtype
        value = x.float()
        mean = value.mean(-1, keepdim=True)
        variance = (value - mean).square().mean(-1, keepdim=True)
        value = (value - mean) * torch.rsqrt(variance + self.eps)
        return (value * self.qk_scale.float()).to(dtype=dtype)


class AdaptiveLayerNorm(nn.Module):
    """Condition an affine-free LayerNorm with learned scale and bias."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_beta = nn.Linear(dim, 2 * dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply adaptive normalization using a shape-aligned condition."""
        gamma, beta = self.gamma_beta(condition).chunk(2, dim=-1)
        return (1.0 + gamma) * self.layer_norm(x) + beta


class SwiGLU(nn.Module):
    """Checkpoint-compatible MIRA SwiGLU feed-forward network."""

    def __init__(self, dim: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.swish_linear = nn.Linear(dim, hidden, bias=False)
        self.gate_linear = nn.Linear(dim, hidden, bias=False)
        self.output_linear = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward projection."""
        return self.output_linear(F.silu(self.swish_linear(x)) * self.gate_linear(x))


class MiraSelfAttention(nn.Module):
    """GQA self-attention backed by FlashDreams native attention and KV cache."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        gating: bool = False,
        backend: Literal["math", "efficient", "cudnn", "flash"] = "cudnn",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.gating = gating
        kv_dim = num_kv_heads * self.head_dim
        self.wqkv = nn.Linear(
            dim, dim + 2 * kv_dim + (dim if gating else 0), bias=False
        )
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_ln = QKLayerNorm(num_heads, self.head_dim)
        self.k_ln = QKLayerNorm(num_kv_heads, self.head_dim)
        self.attention = NativeAttention(qkv_format="bshd", backend=backend)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        repeats = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeats, dim=2) if repeats > 1 else x

    def forward(
        self,
        x: Tensor,
        *,
        rotary: tuple[Tensor, Tensor] | None = None,
        causal: bool = False,
        kv_cache: BlockKVCache | None = None,
        return_kv: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run attention and optionally return raw temporal keys and values."""
        batch, length, _ = x.shape
        kv_dim = self.num_kv_heads * self.head_dim
        pieces = self.wqkv(x).split(
            [self.dim, kv_dim, kv_dim] + ([self.dim] if self.gating else []), dim=-1
        )
        q = self.q_ln(pieces[0].view(batch, length, self.num_heads, self.head_dim))
        k = self.k_ln(pieces[1].view(batch, length, self.num_kv_heads, self.head_dim))
        v = pieces[2].view(batch, length, self.num_kv_heads, self.head_dim)
        gate = pieces[3] if self.gating else None
        raw_kv = (k, v)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.cached_k(), kv_cache.cached_v()
        if rotary is not None:
            q = apply_rotary(q, (rotary[0][-length:], rotary[1][-length:]))
            k = apply_rotary(k, rotary)
        k, v = self._repeat_kv(k), self._repeat_kv(v)

        if causal and kv_cache is None and length > 1:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
            ).transpose(1, 2)
        else:
            out = self.attention(q, k, v)
        out = out.reshape(batch, length, self.dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        out = self.wo(out)
        return (out, raw_kv) if return_kv else out


class LayerScale(nn.Module):
    """Learned residual scale used by the codec decoder."""

    def __init__(self, dim: int, initial: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(initial * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale a residual branch channel-wise."""
        return x * self.gamma
