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

"""FlashDreams transformer contract for native MIRA autoregressive inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor

from flashdreams.infra.compile import compile_module
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)

from mira_integration.network import MiraDiT, MiraDiTCache, MiraDiTConfig


@dataclass(kw_only=True)
class MiraTransformerCache(TransformerAutoregressiveCache):
    """Long-lived MIRA transformer state for one rollout."""

    network_cache: MiraDiTCache
    """Per-temporal-block FlashDreams KV caches."""

    clean_past: Tensor
    """Latest clean latent tokens used by past conditioning."""

    pending_clean: Tensor | None = None
    """Current generated latent committed after the cache-update forward."""

    autoregressive_index: int = -1
    """Current AR index; ``-1`` before generation starts."""

    def start(self, autoregressive_index: int) -> None:
        """Prepare every temporal KV cache for the current AR step."""
        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        """Commit KV bookkeeping and advance clean-past conditioning."""
        self.network_cache.after_update(autoregressive_index)
        assert self.pending_clean is not None
        self.clean_past = self.pending_clean
        self.pending_clean = None


@dataclass(kw_only=True)
class MiraTransformerConfig(TransformerConfig):
    """Config for MIRA's FlashDreams transformer wrapper."""

    _target: type["MiraTransformer"] = field(default_factory=lambda: MiraTransformer)

    network: MiraDiTConfig = field(default_factory=MiraDiTConfig)
    """Checkpoint-compatible MIRA network config."""

    dtype: torch.dtype = torch.bfloat16
    """Parameter and activation dtype used for inference."""

    compile_network: bool = False
    """Compile the checkpoint-compatible network after weight loading."""


class MiraTransformer(Transformer[MiraTransformerCache]):
    """Adapt MIRA's single-latent AR model to FlashDreams diffusion contracts."""

    config: MiraTransformerConfig
    network: MiraDiT

    def __init__(self, config: MiraTransformerConfig) -> None:
        super().__init__(config)
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        ):
            raise RuntimeError("MIRA native inference currently supports one GPU")
        self.config = config
        self.network = config.network.setup().to(dtype=config.dtype).eval()
        self._batch_size: int | None = None
        self._height: int | None = None
        self._width: int | None = None

    def finish_loading(self) -> None:
        """Compile the network after checkpoint restoration when requested."""
        if self.config.compile_network:
            self.network = compile_module(self.network)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return post-patchify shape ``[B,H*W,C]`` for one latent frame."""
        assert (
            self._batch_size is not None
            and self._height is not None
            and self._width is not None
        )
        return (
            self._batch_size,
            self._height * self._width,
            self.config.network.latent_dim,
        )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Flatten video latents or encode a two-row keyboard payload."""
        if x.ndim == 3 and x.shape[-1] == 9:
            return self.network.encode_actions(x)
        assert x.ndim == 5 and x.shape[2] == 1, (
            f"MIRA expects [B,C,1,H,W], got {tuple(x.shape)}"
        )
        return rearrange(x, "b c 1 h w -> b (h w) c")

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Restore flattened current-frame tokens to ``[B,C,1,H,W]``."""
        assert self._height is not None and self._width is not None
        return rearrange(x, "b (h w) c -> b c 1 h w", h=self._height, w=self._width)

    def initialize_autoregressive_cache(
        self,
        *,
        context_latents: Tensor,
        context_action_rows: Tensor,
    ) -> MiraTransformerCache:
        """Prime MIRA's temporal caches from bootstrap latents and actions."""
        assert context_latents.ndim == 5
        batch, channels, _, height, width = context_latents.shape
        cfg = self.config.network
        assert (channels, height, width) == (
            cfg.latent_dim,
            cfg.latent_height,
            cfg.latent_width,
        )
        self._batch_size, self._height, self._width = batch, height, width
        network_cache = self.network.initialize_cache(
            context_latents.to(dtype=self.config.dtype),
            context_action_rows,
        )
        clean_past = rearrange(context_latents[:, :, -1:], "b c 1 h w -> b (h w) c").to(
            dtype=self.config.dtype
        )
        return MiraTransformerCache(
            network_cache=network_cache,
            clean_past=clean_past,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: MiraTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Predict current-frame flow using action and clean-past conditioning."""
        assert input is not None, "MIRA requires encoded keyboard actions"
        return self.network(
            noisy_latent,
            timesteps=timestep,
            cache=cache.network_cache,
            action_embedding=input,
            clean_past=cache.clean_past,
        )

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: MiraTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Stage the clean latent for next-step past conditioning."""
        _ = input
        cache.pending_clean = clean_latent.detach()
        return clean_latent
