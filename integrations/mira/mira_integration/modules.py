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

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl

from flashdreams.core.attention import BlockKVCache, NativeAttention


@triton.jit
def _mira_rotary_kernel(
    x_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_fs,
    stride_fd,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % H
    token = (pid // H) % S
    batch = pid // (H * S)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    pair_d = tl.where((d & 1) == 0, d + 1, d - 1)

    x_base = x_ptr + batch * stride_xb + token * stride_xs + head * stride_xh
    out_base = out_ptr + batch * stride_ob + token * stride_os + head * stride_oh
    freq_base = token * stride_fs

    raw_value = tl.load(x_base + d * stride_xd, mask=mask, other=0.0)
    value = raw_value.to(tl.float32)
    pair = tl.load(
        x_base + pair_d * stride_xd,
        mask=mask & (pair_d < D),
        other=0.0,
    ).to(tl.float32)
    cos = tl.load(cos_ptr + freq_base + d * stride_fd, mask=mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + freq_base + d * stride_fd, mask=mask, other=0.0).to(
        tl.float32
    )
    rotated = tl.where((d & 1) == 0, -pair, pair)
    tl.store(out_base + d * stride_od, value * cos + rotated * sin, mask=mask)

@triton.jit
def _mira_qk_norm_rope_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_sh,
    stride_sd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_fs,
    stride_fd,
    eps: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_ROTARY: tl.constexpr,
    NORM_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % H
    token = (pid // H) % S
    batch = pid // (H * S)
    d = tl.arange(0, BLOCK_D)
    mask = d < D

    x_base = x_ptr + batch * stride_xb + token * stride_xs + head * stride_xh
    scale_base = scale_ptr + head * stride_sh
    out_base = out_ptr + batch * stride_ob + token * stride_os + head * stride_oh

    raw_value = tl.load(x_base + d * stride_xd, mask=mask, other=0.0)
    value = raw_value.to(tl.float32)
    mean = tl.sum(tl.where(mask, value, 0.0), axis=0) / D
    centered = value - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / D
    scale = tl.load(scale_base + d * stride_sd, mask=mask, other=0.0).to(
        tl.float32
    )
    normalized = centered * tl.rsqrt(variance + eps) * scale

    if APPLY_ROTARY:
        pair_d = tl.where((d & 1) == 0, d + 1, d - 1)
        pair_raw_value = tl.load(
            x_base + pair_d * stride_xd,
            mask=mask & (pair_d < D),
            other=0.0,
        )
        pair_value = pair_raw_value.to(tl.float32)
        pair_scale = tl.load(
            scale_base + pair_d * stride_sd,
            mask=mask & (pair_d < D),
            other=0.0,
        ).to(tl.float32)
        pair_normalized = (pair_value - mean) * tl.rsqrt(variance + eps) * pair_scale
        if NORM_DTYPE == 1:
            normalized = normalized.to(tl.float16).to(tl.float32)
            pair_normalized = pair_normalized.to(tl.float16).to(tl.float32)
        if NORM_DTYPE == 2:
            normalized = normalized.to(tl.bfloat16).to(tl.float32)
            pair_normalized = pair_normalized.to(tl.bfloat16).to(tl.float32)
        freq_base = token * stride_fs
        cos = tl.load(
            cos_ptr + freq_base + d * stride_fd,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + freq_base + d * stride_fd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where((d & 1) == 0, -pair_normalized, pair_normalized)
        normalized = normalized * cos + rotated * sin

    tl.store(out_base + d * stride_od, normalized, mask=mask)


@triton.jit
def _mira_swiglu_kernel(
    swish_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    swish = tl.load(swish_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = (swish / (1.0 + tl.exp(-swish))) * gate
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _mira_tiny_causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    query_pos = pid % S
    head = (pid // S) % H
    batch = pid // (S * H)

    d = tl.arange(0, BLOCK_D)
    s = tl.arange(0, BLOCK_S)
    d_mask = d < D
    s_mask = s < S
    causal_mask = s <= query_pos

    q_base = q_ptr + batch * stride_qb + query_pos * stride_qs + head * stride_qh
    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh
    out_base = out_ptr + batch * stride_ob + query_pos * stride_os + head * stride_oh

    q = tl.load(q_base + d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
    k = tl.load(
        k_base + s[:, None] * stride_ks + d[None, :] * stride_kd,
        mask=s_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    scores = tl.where(s_mask & causal_mask, scores, -float("inf"))
    scores = scores - tl.max(scores, axis=0)
    weights = tl.exp(scores)
    weights = weights / tl.sum(weights, axis=0)

    v = tl.load(
        v_base + s[:, None] * stride_vs + d[None, :] * stride_vd,
        mask=s_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(weights[:, None] * v, axis=0)
    tl.store(out_base + d * stride_od, out, mask=d_mask)


def _triton_block_d(head_dim: int) -> int:
    assert triton is not None
    return max(int(triton.next_power_of_2(head_dim)), 16)


def _triton_block_s(length: int) -> int:
    assert triton is not None
    return max(int(triton.next_power_of_2(length)), 2)


def _apply_rotary_cuda(x: Tensor, frequencies: tuple[Tensor, Tensor]) -> Tensor | None:
    if triton is None or not x.is_cuda:
        return None
    if x.ndim != 4:
        return None
    cos, sin = frequencies
    batch, length, heads, head_dim = x.shape
    if cos.shape != (length, head_dim) or sin.shape != (length, head_dim):
        return None
    out = torch.empty_like(x)
    _mira_rotary_kernel[(batch * length * heads,)](
        x,
        out,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        cos.stride(0),
        cos.stride(1),
        S=length,
        H=heads,
        D=head_dim,
        BLOCK_D=_triton_block_d(head_dim),
        num_warps=4,
        num_stages=2,
    )
    return out


def _qk_layer_norm_cuda(
    x: Tensor,
    scale: Tensor,
    eps: float,
    rotary: tuple[Tensor, Tensor] | None = None,
) -> Tensor | None:
    if triton is None or not x.is_cuda:
        return None
    if x.ndim != 4:
        return None
    batch, length, heads, head_dim = x.shape
    if scale.shape != (heads, head_dim):
        return None
    cos: Tensor
    sin: Tensor
    if rotary is None:
        cos = x.new_empty((0, 0))
        sin = x.new_empty((0, 0))
        apply_rotary_frequencies = False
    else:
        cos, sin = rotary
        if cos.shape != (length, head_dim) or sin.shape != (length, head_dim):
            return None
        apply_rotary_frequencies = True
    out = torch.empty_like(x)
    norm_dtype = 1 if x.dtype is torch.float16 else 2 if x.dtype is torch.bfloat16 else 0
    _mira_qk_norm_rope_kernel[(batch * length * heads,)](
        x,
        scale,
        out,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        cos.stride(0) if apply_rotary_frequencies else 0,
        cos.stride(1) if apply_rotary_frequencies else 0,
        eps,
        S=length,
        H=heads,
        D=head_dim,
        BLOCK_D=_triton_block_d(head_dim),
        APPLY_ROTARY=apply_rotary_frequencies,
        NORM_DTYPE=norm_dtype,
        num_warps=4,
        num_stages=2,
    )
    return out


def _tiny_causal_attention_cuda(q: Tensor, k: Tensor, v: Tensor) -> Tensor | None:
    if triton is None or not q.is_cuda:
        return None
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        return None
    batch, length, heads, head_dim = q.shape
    if length < 2 or length > 8 or head_dim > 128:
        return None
    out = torch.empty_like(q)
    _mira_tiny_causal_attention_kernel[(batch * heads * length,)](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        S=length,
        H=heads,
        D=head_dim,
        BLOCK_S=_triton_block_s(length),
        BLOCK_D=_triton_block_d(head_dim),
        SCALE=head_dim**-0.5,
        num_warps=4,
        num_stages=2,
    )
    return out


def _swiglu_gate_cuda(swish: Tensor, gate: Tensor) -> Tensor | None:
    if triton is None or not swish.is_cuda:
        return None
    if swish.shape != gate.shape:
        return None
    if not swish.is_contiguous() or not gate.is_contiguous():
        return None
    out = torch.empty_like(swish)
    n_elements = out.numel()
    block_size = 256
    _mira_swiglu_kernel[(triton.cdiv(n_elements, block_size),)](
        swish,
        gate,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=4,
    )
    return out


@nvtx.annotate("mira.modules.qk_layer_norm")
def qk_layer_norm(x: Tensor, scale: Tensor, eps: float) -> Tensor:
    fused = _qk_layer_norm_cuda(x, scale, eps)
    if fused is not None:
        return fused
    dtype = x.dtype
    value = x.float()
    mean = value.mean(-1, keepdim=True)
    variance = (value - mean).square().mean(-1, keepdim=True)
    value = (value - mean) * torch.rsqrt(variance + eps)
    return (value * scale.float()).to(dtype=dtype)


@nvtx.annotate("mira.modules.qk_layer_norm_rotary")
def qk_layer_norm_rotary(
    x: Tensor,
    scale: Tensor,
    eps: float,
    frequencies: tuple[Tensor, Tensor],
) -> Tensor:
    fused = _qk_layer_norm_cuda(x, scale, eps, frequencies)
    if fused is not None:
        return fused
    return apply_rotary(qk_layer_norm(x, scale, eps), frequencies)


@nvtx.annotate("mira.modules.apply_rotary")
def apply_rotary(x: Tensor, frequencies: tuple[Tensor, Tensor]) -> Tensor:
    """Apply pairwise rotary frequencies to ``[B, S, H, D]`` attention states."""
    fused = _apply_rotary_cuda(x, frequencies)
    if fused is not None:
        return fused
    cos, sin = (frequency.unsqueeze(-2) for frequency in frequencies)
    paired = x.float().unflatten(-1, (-1, 2))
    real, imag = paired.unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(-2)
    return (x.float() * cos + rotated * sin).to(dtype=x.dtype)


@nvtx.annotate("mira.modules.temporal_rope")
def temporal_rope(
    length: int, head_dim: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build MIRA's seconds-scaled temporal RoPE table."""
    positions = torch.arange(length, dtype=torch.float32, device=device) / 10.0
    bands = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
    inv_freq = torch.exp(bands * (-math.log(64.0) * 2.0 / head_dim))
    phase = positions[:, None] * inv_freq[None, :]
    return phase.cos().repeat_interleave(2, -1), phase.sin().repeat_interleave(2, -1)


@nvtx.annotate("mira.modules.spatial_rope")
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


@nvtx.annotate("mira.modules.decoder_rope")
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

    @nvtx.annotate("QKLayerNorm.forward")
    def forward(self, x: Tensor) -> Tensor:
        """Normalize the final channel in fp32 and restore the input dtype."""
        return qk_layer_norm(x, self.qk_scale, self.eps)


class AdaptiveLayerNorm(nn.Module):
    """Condition an affine-free LayerNorm with learned scale and bias."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_beta = nn.Linear(dim, 2 * dim)

    @nvtx.annotate("AdaptiveLayerNorm.forward")
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

    @nvtx.annotate("SwiGLU.forward")
    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward projection."""
        swish = self.swish_linear(x)
        gate = self.gate_linear(x)
        gated = _swiglu_gate_cuda(swish, gate)
        if gated is None:
            gated = F.silu(swish) * gate
        return self.output_linear(gated)


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
        causal_attention_backend: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.gating = gating
        self.causal_attention_backend = causal_attention_backend
        kv_dim = num_kv_heads * self.head_dim
        self.wqkv = nn.Linear(
            dim, dim + 2 * kv_dim + (dim if gating else 0), bias=False
        )
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_ln = QKLayerNorm(num_heads, self.head_dim)
        self.k_ln = QKLayerNorm(num_kv_heads, self.head_dim)
        self.attention = NativeAttention(qkv_format="bshd", backend=backend)

    @nvtx.annotate("MiraSelfAttention._repeat_kv")
    def _repeat_kv(self, x: Tensor) -> Tensor:
        repeats = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeats, dim=2) if repeats > 1 else x

    @nvtx.annotate("MiraSelfAttention.forward")
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
        with nvtx.annotate("MiraSelfAttention.wqkv"):
            pieces = self.wqkv(x).split(
                [self.dim, kv_dim, kv_dim] + ([self.dim] if self.gating else []),
                dim=-1,
            )
        q_in = pieces[0].view(batch, length, self.num_heads, self.head_dim)
        k_in = pieces[1].view(batch, length, self.num_kv_heads, self.head_dim)
        q_rotary = (
            None
            if rotary is None
            else (rotary[0][-length:], rotary[1][-length:])
        )
        can_fuse_k_rotary = rotary is not None and kv_cache is None and not return_kv
        with nvtx.annotate("MiraSelfAttention.q_norm_rotary"):
            q = (
                self.q_ln(q_in)
                if q_rotary is None
                else qk_layer_norm_rotary(
                    q_in,
                    self.q_ln.qk_scale,
                    self.q_ln.eps,
                    q_rotary,
                )
            )
        with nvtx.annotate("MiraSelfAttention.k_norm_rotary"):
            k = (
                qk_layer_norm_rotary(
                    k_in,
                    self.k_ln.qk_scale,
                    self.k_ln.eps,
                    rotary,
                )
                if can_fuse_k_rotary
                else self.k_ln(k_in)
            )
        with nvtx.annotate("MiraSelfAttention.value_view"):
            v = pieces[2].view(batch, length, self.num_kv_heads, self.head_dim)
            gate = pieces[3] if self.gating else None
            raw_kv = (k, v)

        with nvtx.annotate("MiraSelfAttention.kv_cache"):
            if kv_cache is not None:
                kv_cache.update(k, v)
                k, v = kv_cache.cached_k(), kv_cache.cached_v()
        with nvtx.annotate("MiraSelfAttention.rotary"):
            if rotary is not None:
                if q_rotary is None:
                    with nvtx.annotate("MiraSelfAttention.rotary.q"):
                        q = apply_rotary(q, (rotary[0][-length:], rotary[1][-length:]))
                if not can_fuse_k_rotary:
                    with nvtx.annotate("MiraSelfAttention.rotary.k"):
                        k = apply_rotary(k, rotary)
        with nvtx.annotate("MiraSelfAttention.repeat_kv"):
            k, v = self._repeat_kv(k), self._repeat_kv(v)

        with nvtx.annotate("MiraSelfAttention.attention"):
            if causal and kv_cache is None and length > 1:
                if self.causal_attention_backend == "triton":
                    out = _tiny_causal_attention_cuda(q, k, v)
                    if out is None:
                        out = F.scaled_dot_product_attention(
                            q.transpose(1, 2),
                            k.transpose(1, 2),
                            v.transpose(1, 2),
                            is_causal=True,
                        ).transpose(1, 2)
                else:
                    out = F.scaled_dot_product_attention(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        is_causal=True,
                    ).transpose(1, 2)
            else:
                out = self.attention(q, k, v)
        with nvtx.annotate("MiraSelfAttention.output_projection"):
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

    @nvtx.annotate("LayerScale.forward")
    def forward(self, x: Tensor) -> Tensor:
        """Scale a residual branch channel-wise."""
        return x * self.gamma

    @nvtx.annotate("LayerScale.residual")
    def residual(self, residual: Tensor, branch: Tensor) -> Tensor:
        """Add a scaled residual branch using one fused multiply-add op."""
        return torch.addcmul(residual, branch, self.gamma)
