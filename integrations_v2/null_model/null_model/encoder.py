# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar input encoder for the deterministic NULL model."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)


@dataclass(kw_only=True)
class NullInputEncoderConfig(EncoderConfig):
    """Config selecting the NULL model's per-step scalar encoder."""

    _target: type["NullInputEncoder"] = field(default_factory=lambda: NullInputEncoder)


class NullInputEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Stateless per-step encoder of model inputs.

    The pipeline's `encoder` slot requires
    `flashdreams.infra.encoder.StreamingEncoder` because we decided to allow the null encoder to ingest an input 
    every AR step. This reference encoder is stateless as it does not use its encoder cache.
    """

    def initialize_autoregressive_cache(self, **_context: Any) -> StreamingEncoderCache:
        """Return an empty cache for a stateless encoder.

        The pipeline calls this once when it initializes a rollout. Real
        control encoders can consume `_context` and return a cache subclass
        carrying state between AR steps.
        """
        return StreamingEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Return a `[1, 1]` input tensor unchanged.

        Args:
            input: Tesor with shape `[1, 1]`.
            autoregressive_index: Unused because
                this encoder has no step-dependent behavior.
            cache: Unused.

        Returns:
            The validated input tensor.

        Raises:
            AssertionError: `input` is not a tensor with shape `[1, 1]`.
        """
        del autoregressive_index, cache

        assert isinstance(input, Tensor), (
            f"expected input to be a Tensor, got {type(input).__name__}"
        )
        assert input.shape == (1, 1), (
            f"expected input tensor shape (1, 1), got {tuple(input.shape)}"
        )
        return input
