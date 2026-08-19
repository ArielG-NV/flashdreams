# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic RGB transformer for the NULL model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)


@dataclass(kw_only=True)
class NullTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for the NULL transformer."""

    autoregressive_index: int = -1
    """Current AR step; ``-1`` before the first :meth:`start` call."""

    def start(self, autoregressive_index: int) -> None:
        """Record the AR step that determines the output value.

        Args:
            autoregressive_index: Current zero-based AR step.
        """
        self.autoregressive_index = autoregressive_index


@dataclass(kw_only=True)
class NullTransformerConfig(TransformerConfig):
    """Config for the deterministic NULL transformer."""

    _target: type["NullTransformer"] = field(default_factory=lambda: NullTransformer)


class NullTransformer(Transformer[NullTransformerCache]):
    """Emit constant RGB chunks whose value equals the current AR index."""

    def __init__(self, config: NullTransformerConfig) -> None:
        super().__init__(config)
        self._device_anchor = torch.nn.Parameter(
            torch.zeros(()),
            requires_grad=False,
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return output shape ``[B, C=3, T, H, W]``."""
        return (1, 3, 1, 1, 1)

    def initialize_autoregressive_cache(self) -> NullTransformerCache:
        """Return a fresh cache for a deterministic rollout."""
        return NullTransformerCache()

    def initial_noise(
        self,
        *,
        latent_shape: tuple[int, ...],
        rng: torch.Generator | None,
        cache: NullTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Return zeros so the RGB result is exact."""
        del rng, cache, input
        return torch.zeros(latent_shape, device=self.device, dtype=self.dtype)

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: NullTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Predict flow from random noise to the current AR value."""
        del timestep, input
        return noisy_latent - cache.autoregressive_index

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """Return ``x`` unchanged."""
        return x

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Return ``x`` unchanged."""
        return x
