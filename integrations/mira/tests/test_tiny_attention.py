# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for MIRA's tiny temporal attention kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mira_integration.decoder import MiraDecoderConfig
from mira_integration.modules import (
    _swiglu_gate_cuda,
    _tiny_causal_attention_cuda,
)

pytestmark = pytest.mark.ci_gpu


def test_tiny_causal_attention_cuda_matches_sdpa() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    q = torch.randn(7, 3, 4, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(7, 3, 4, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(7, 3, 4, 16, device="cuda", dtype=torch.bfloat16)

    actual = _tiny_causal_attention_cuda(q, k, v)
    assert actual is not None
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_swiglu_gate_cuda_matches_torch() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(1)
    swish = torch.randn(17, 23, 64, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(swish)

    actual = _swiglu_gate_cuda(swish, gate)
    assert actual is not None
    expected = F.silu(swish) * gate

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_decoder_cuda_graph_matches_eager_decoder() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(3)
    reference = MiraDecoderConfig(
        latent_dim=4,
        width=32,
        depth=2,
        num_heads=4,
        patch_size=2,
        dtype=torch.float32,
        attention_backend="math",
        causal_temporal_attention_backend="torch",
        use_cuda_graph=False,
    ).setup().cuda()
    candidate = MiraDecoderConfig(
        latent_dim=4,
        width=32,
        depth=2,
        num_heads=4,
        patch_size=2,
        dtype=torch.float32,
        attention_backend="math",
        causal_temporal_attention_backend="triton",
        use_cuda_graph=True,
        cuda_graph_warmup_iters=1,
    ).setup().cuda()
    candidate.load_state_dict(reference.state_dict())

    context = torch.randn(1, 4, 2, 2, 2, device="cuda")
    inputs = [torch.randn(1, 4, 1, 2, 2, device="cuda") for _ in range(3)]
    reference_cache = reference.initialize_autoregressive_cache(context_latents=context)
    candidate_cache = candidate.initialize_autoregressive_cache(context_latents=context)

    for input in inputs:
        expected = reference(input, cache=reference_cache)
        actual = candidate(input, cache=candidate_cache)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
