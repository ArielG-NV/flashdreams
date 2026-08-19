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

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingDecoderCache | None = None,
    ) -> Tensor:
        """Return ``input`` unchanged."""
        _ = autoregressive_index, cache
        return input
