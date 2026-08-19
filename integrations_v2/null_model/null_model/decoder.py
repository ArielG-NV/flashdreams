# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity output decoder for the deterministic NULL model."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCache,
)


@dataclass(kw_only=True)
class NullDecoderConfig(DecoderConfig):
    """Config for the NULL model identity decoder."""

    _target: type["NullDecoder"] = field(default_factory=lambda: NullDecoder)


class NullDecoder(StreamingDecoder[StreamingDecoderCache]):
    """Pass generated output through unchanged."""

    def initialize_autoregressive_cache(
        self, **_context: Any
    ) -> StreamingDecoderCache:
        """Return an empty per-rollout cache."""
        return StreamingDecoderCache()

    # Ingest an `input` tensor and return it unchanged.
    # This effectively means that decoding is a no-op.
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingDecoderCache | None = None,
    ) -> Tensor:
        """Return ``input`` unchanged."""
        del autoregressive_index, cache
        return input
