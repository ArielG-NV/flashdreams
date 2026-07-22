# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for MIRA's tiny temporal attention kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mira_integration.modules import _tiny_causal_attention_cuda

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
