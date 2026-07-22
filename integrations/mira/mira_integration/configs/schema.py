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

"""Configuration contracts for MIRA models, controls, and WebRTC serving."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mira_integration.pipeline import MiraPipelineConfig


@dataclass(frozen=True, slots=True)
class MiraInputBinding:
    """Map one normalized browser input to one checkpoint action key."""

    browser_key: str
    """Normalized key sent by the browser."""

    checkpoint_key: str
    """Key name expected by the checkpoint action encoder."""

    label: str
    """Short key-cap label shown in the browser."""

    action: str
    """Human-readable action name shown beside the key."""

    group: str
    """Control-group heading used by the browser."""

    group_description: str
    """Control-group description used by the browser."""

    aliases: tuple[str, ...] = ()
    """Additional normalized ``KeyboardEvent.key`` or ``code`` aliases."""

    def to_public_dict(self) -> dict[str, Any]:
        """Return the browser-safe representation of this binding."""
        return {
            "key": self.browser_key,
            "checkpointKey": self.checkpoint_key,
            "label": self.label,
            "action": self.action,
            "group": self.group,
            "groupDescription": self.group_description,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class MiraModelMetadata:
    """Describe checkpoint identity, controls, and player layout."""

    name: str
    """Stable configuration slug accepted by ``mira-webrtc``."""

    display_name: str
    """Human-readable model name shown in telemetry."""

    checkpoint: str
    """Hugging Face repository or local checkpoint-bundle identifier."""

    player_count: int
    """Number of synchronized player views produced by the checkpoint."""

    steps: int
    """Default sampler steps used for each generated latent frame."""

    latent_height: int
    """Latent grid height for one player view."""

    latent_width: int
    """Latent grid width for one player view."""

    video_width: int
    """Pixel width of one decoded player view."""

    video_height: int
    """Pixel height of one decoded player view."""

    frames_per_chunk: int
    """Pixel frames emitted by one autoregressive step."""

    input_key_map: tuple[MiraInputBinding, ...]
    """Ordered browser-to-checkpoint control map."""

    @property
    def browser_keys(self) -> frozenset[str]:
        """Return normalized browser keys accepted by this model."""
        return frozenset(binding.browser_key for binding in self.input_key_map)

    @property
    def num_action_keys(self) -> int:
        """Return the checkpoint action row width described by the input map."""
        return len(self.input_key_map)

    def checkpoint_keys(self, keys: frozenset[str]) -> list[str]:
        """Translate held browser keys in checkpoint vocabulary order."""
        return [
            binding.checkpoint_key
            for binding in self.input_key_map
            if binding.browser_key in keys
        ]

    def to_public_dict(self) -> dict[str, Any]:
        """Return model metadata used to construct the browser UI."""
        rows, columns = preview_grid_dimensions(self.player_count)
        return {
            "name": self.name,
            "displayName": self.display_name,
            "checkpoint": self.checkpoint,
            "playerCount": self.player_count,
            "steps": self.steps,
            "framesPerChunk": self.frames_per_chunk,
            "inputs": [binding.to_public_dict() for binding in self.input_key_map],
            "latent": {
                "height": self.latent_height,
                "width": self.latent_width,
            },
            "video": {
                "width": self.video_width,
                "height": self.video_height,
            },
            "previewGrid": {"rows": rows, "columns": columns},
        }


@dataclass(frozen=True, slots=True)
class MiraWebRTCModelConfig:
    """Bind public MIRA metadata to its native inference pipeline."""

    metadata: MiraModelMetadata
    """Checkpoint, controls, and player-layout metadata."""

    pipeline: MiraPipelineConfig
    """Native pipeline literal matching ``metadata.checkpoint``."""


@dataclass(frozen=True, slots=True)
class MiraManifest:
    """Validated input maps and demos loaded from a MIRA manifest."""

    input_maps: dict[str, tuple[MiraInputBinding, ...]]
    """Ordered control bindings keyed by manifest input-map identifier."""

    demos: dict[str, MiraModelMetadata]
    """Model metadata keyed by command-line demo name."""


def preview_grid_dimensions(player_count: int) -> tuple[int, int]:
    """Return a compact near-square ``(rows, columns)`` player grid.

    Args:
        player_count: Positive number of player views.

    Returns:
        Grid dimensions with enough cells for every player.

    Raises:
        ValueError: ``player_count`` is not positive.
    """
    if player_count <= 0:
        raise ValueError("player_count must be > 0")
    columns = math.ceil(math.sqrt(player_count))
    rows = math.ceil(player_count / columns)
    return rows, columns


__all__ = [
    "MiraInputBinding",
    "MiraManifest",
    "MiraModelMetadata",
    "MiraWebRTCModelConfig",
    "preview_grid_dimensions",
]
