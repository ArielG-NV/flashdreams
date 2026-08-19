# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity input encoder for the deterministic NULL model."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)


@dataclass(kw_only=True)
class NullInputEncoderConfig(EncoderConfig):
    """Config for the NULL model identity input encoder."""

    _target: type["NullInputEncoder"] = field(
        default_factory=lambda: NullInputEncoder
    )


class NullInputEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Pass per-step input through unchanged."""

    def initialize_autoregressive_cache(
        self, **_context: Any
    ) -> StreamingEncoderCache:
        """Return an empty per-rollout cache."""
        return StreamingEncoderCache()

    def forward(
        self,
        input: Any,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Return a dummy input tensor."""
        _ = input, autoregressive_index, cache
        return torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
