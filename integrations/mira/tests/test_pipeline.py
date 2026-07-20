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

"""CPU contract tests for the FlashDreams-native MIRA components."""

from __future__ import annotations

import pytest
import torch

from mira_integration.decoder import MiraDecoderConfig
from mira_integration.encoder import MiraControlEncoderConfig
from mira_integration.network import MiraDiTConfig
from mira_integration.scheduler import MiraFlowSchedulerConfig
from mira_integration.transformer import MiraTransformerConfig

pytestmark = pytest.mark.ci_cpu


def test_control_encoder_preserves_two_row_alignment() -> None:
    encoder = MiraControlEncoderConfig().setup()
    previous = torch.zeros(1, 1, 9, dtype=torch.int32)
    previous[..., 2] = 1
    cache = encoder.initialize_autoregressive_cache(previous_row=previous)
    first = encoder(["W", "D"], cache=cache)
    second = encoder([], autoregressive_index=1, cache=cache)
    assert torch.equal(first[:, :1], previous)
    assert first[0, 1].tolist() == [1, 0, 0, 1, 0, 0, 0, 0, 0]
    assert torch.equal(second[:, :1], first[:, 1:])
    assert torch.count_nonzero(second[:, 1:]) == 0


def test_flow_scheduler_integrates_in_increasing_time() -> None:
    scheduler = MiraFlowSchedulerConfig(
        num_inference_steps=4, schedule_type="linear"
    ).setup()
    initial = torch.zeros(1, 2, 3)
    result = scheduler.sample(initial, lambda sample, tau: torch.ones_like(sample))
    assert torch.equal(result, torch.ones_like(initial))


def _small_transformer_config() -> MiraTransformerConfig:
    return MiraTransformerConfig(
        dtype=torch.float32,
        network=MiraDiTConfig(
            latent_dim=4,
            hidden_dim=32,
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,
            time_attention_every=1,
            latent_height=2,
            latent_width=2,
            attention_gating=True,
            ada_attention=True,
            attention_backend="math",
        ),
    )


def test_native_transformer_primes_and_advances_flashdreams_cache() -> None:
    transformer = _small_transformer_config().setup().eval()
    context = torch.randn(1, 4, 3, 2, 2)
    action_rows = torch.zeros(1, 4, 9, dtype=torch.int32)
    cache = transformer.initialize_autoregressive_cache(
        context_latents=context, context_action_rows=action_rows
    )
    encoded_action = transformer.patchify_and_maybe_split_cp(
        torch.zeros(1, 2, 9, dtype=torch.int32)
    )
    noisy = torch.randn(transformer.latent_shape)
    cache.start(0)
    flow = transformer.predict_flow(
        noisy, torch.tensor(0.5), cache, input=encoded_action
    )
    transformer.postprocess_clean_latent(noisy, cache)
    transformer.predict_flow(noisy, torch.tensor(0.8), cache, input=encoded_action)
    cache.finalize(0)
    assert flow.shape == transformer.latent_shape
    assert torch.equal(cache.clean_past, noisy)
    assert all(item is None or item.size == 4 for item in cache.network_cache.temporal)


def test_native_decoder_emits_two_rgb_frames() -> None:
    decoder = MiraDecoderConfig(
        latent_dim=4,
        width=32,
        depth=2,
        num_heads=4,
        patch_size=2,
        dtype=torch.float32,
        attention_backend="math",
    ).setup()
    context = torch.randn(1, 4, 2, 2, 2)
    cache = decoder.initialize_autoregressive_cache(context_latents=context)
    output = decoder(torch.randn(1, 4, 1, 2, 2), cache=cache)
    assert output.shape == (1, 2, 3, 8, 8)
    assert output.min() >= 0
    assert output.max() <= 1
