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
    """Config for the NULL model scalar input encoder."""

    _target: type["NullInputEncoder"] = field(
        default_factory=lambda: NullInputEncoder
    )


class NullInputEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Validate and pass through one scalar per-step input."""

    def initialize_autoregressive_cache(
        self, **_context: Any
    ) -> StreamingEncoderCache:
        """Return an empty per-rollout cache."""
        return StreamingEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Return a ``[1, 1]`` input tensor unchanged.

        Args:
            input: Scalar batch input with shape ``[1, 1]``.
            autoregressive_index: Current zero-based AR step.
            cache: Stateless per-rollout encoder cache.

        Returns:
            The validated input tensor.

        Raises:
            AssertionError: ``input`` is not a tensor with shape ``[1, 1]``.
        """
        del autoregressive_index, cache
        assert isinstance(input, Tensor), (
            f"expected input to be a Tensor, got {type(input).__name__}"
        )
        assert input.shape == (1, 1), (
            f"expected input tensor shape (1, 1), got {tuple(input.shape)}"
        )
        return input
