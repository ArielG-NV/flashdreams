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

"""Checkpoint-compatible learned keyboard and mouse action conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import nvtx
import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MiraActionInput:
    """Raw aligned action rows and explicit per-player autopilot state."""

    rows: Tensor
    """Keyboard rows ``[B, T, K]`` containing only binary values."""

    autopilot_mask: Tensor
    """Boolean player mask ``[B]`` selecting learned action dropout."""


@nvtx.annotate("mira.action._symlog_normalize")
def _symlog_normalize(value: Tensor, maximum: float = 2048.0) -> Tensor:
    """Normalize signed mouse deltas with MIRA's logarithmic transform."""
    signed = torch.sign(value) * torch.log1p(torch.abs(value))
    scale = torch.log1p(torch.tensor(maximum, device=value.device))
    return signed / scale


class MiraActionEncoder(nn.Module):
    """Embed two-frame keyboard/mouse action windows into latent-frame tokens."""

    def __init__(
        self,
        num_keys: int = 9,
        dim: int = 2048,
        temporal_downsampling: int = 2,
        per_player_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temporal_downsampling = temporal_downsampling
        mouse_dim = dim // 2
        keyboard_dim = dim - mouse_dim
        split_dim = 2 ** math.floor(math.log2(keyboard_dim / num_keys))
        remainder = keyboard_dim - num_keys * split_dim

        self.mouse_mlp = nn.Linear(2, mouse_dim)
        self.mouse_sensitivity_mlp = nn.Linear(1, mouse_dim)
        self.mouse_sensitivity_dropout_token = nn.Parameter(
            torch.empty(1, 1, mouse_dim)
        )
        self.keyboard_embedding_dict = nn.ModuleDict(
            {str(index): nn.Embedding(2, split_dim) for index in range(num_keys)}
        )
        self.register_buffer(
            "keyboard_zero_vector",
            torch.zeros(1, 1, remainder),
            persistent=False,
        )
        self.keyboard_mlp = nn.Linear(keyboard_dim, keyboard_dim)
        self.mouse_temporal_pool = nn.Linear(
            temporal_downsampling * mouse_dim, mouse_dim
        )
        self.keyboard_temporal_pool = nn.Linear(
            temporal_downsampling * keyboard_dim, keyboard_dim
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.mouse_dropout_token = nn.Parameter(torch.empty(1, 1, mouse_dim))
        if per_player_dropout:
            self.key_dropout_embed = nn.Parameter(torch.empty(num_keys, split_dim))
            self.register_parameter("keyboard_dropout_token", None)
        else:
            self.register_parameter("key_dropout_embed", None)
            self.keyboard_dropout_token = nn.Parameter(torch.empty(1, 1, keyboard_dim))
        self.initial_action_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.mouse_sensitivity_dropout_token, std=0.02)
        nn.init.normal_(self.mouse_dropout_token, std=0.02)
        if self.key_dropout_embed is not None:
            nn.init.normal_(self.key_dropout_embed, std=0.02)
        if self.keyboard_dropout_token is not None:
            nn.init.normal_(self.keyboard_dropout_token, std=0.02)
        nn.init.normal_(self.initial_action_token, std=0.02)

    @nvtx.annotate("MiraActionEncoder.forward")
    def forward(
        self,
        key_presses: Tensor,
        *,
        drop_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode keyboard rows ``[B,T,K]`` into action tokens."""
        batch, steps, _ = key_presses.shape
        assert steps % self.temporal_downsampling == 0, (
            f"action row count {steps} must be divisible by "
            f"{self.temporal_downsampling}"
        )
        if drop_mask is None:
            drop_mask = torch.zeros(
                batch,
                device=key_presses.device,
                dtype=torch.bool,
            )
        if drop_mask.shape != (batch,) or drop_mask.dtype != torch.bool:
            raise ValueError(
                "drop_mask must be a bool tensor with shape "
                f"({batch},), got {drop_mask.dtype} {tuple(drop_mask.shape)}"
            )
        if ((key_presses < 0) | (key_presses > 1)).any():
            raise ValueError("MIRA key presses must contain only binary values")
        dtype = self.mouse_mlp.weight.dtype
        mouse = torch.zeros(batch, steps, 2, device=key_presses.device, dtype=dtype)
        mouse = self.mouse_mlp(_symlog_normalize(mouse))
        sensitivity = torch.ones(batch, 1, 1, device=key_presses.device, dtype=dtype)
        sensitivity = self.mouse_sensitivity_mlp(sensitivity)
        sensitivity = self.mouse_sensitivity_dropout_token.to(dtype=dtype).expand_as(
            sensitivity
        )
        mouse = mouse + sensitivity

        keyboard_parts = [
            self._embed_key(
                key_presses=key_presses,
                key_index=index,
                drop_mask=drop_mask,
            )
            for index in range(key_presses.shape[-1])
        ]
        keyboard_zero_vector = self.get_buffer("keyboard_zero_vector")
        if keyboard_zero_vector.shape[-1] > 0:
            keyboard_parts.append(
                keyboard_zero_vector.expand(batch, steps, -1).to(dtype=dtype)
            )
        keyboard = self.keyboard_mlp(torch.cat(keyboard_parts, dim=-1))

        stride = self.temporal_downsampling
        mouse = self.mouse_temporal_pool(mouse.unflatten(1, (-1, stride)).flatten(2))
        keyboard = self.keyboard_temporal_pool(
            keyboard.unflatten(1, (-1, stride)).flatten(2)
        )
        if drop_mask.any():
            mask = drop_mask[:, None, None]
            mouse = torch.where(
                mask,
                self.mouse_dropout_token.to(dtype=mouse.dtype),
                mouse,
            )
            if self.keyboard_dropout_token is not None:
                keyboard = torch.where(
                    mask,
                    self.keyboard_dropout_token.to(dtype=keyboard.dtype),
                    keyboard,
                )
        encoded = self.joint_mlp(torch.cat((mouse, keyboard), dim=-1))
        initial = self.initial_action_token.expand(batch, -1, -1).to(
            dtype=encoded.dtype
        )
        return torch.cat((initial, encoded), dim=1)

    @nvtx.annotate("MiraActionEncoder._embed_key")
    def _embed_key(
        self,
        *,
        key_presses: Tensor,
        key_index: int,
        drop_mask: Tensor,
    ) -> Tensor:
        embedded = self.keyboard_embedding_dict[str(key_index)](
            key_presses[..., key_index].to(dtype=torch.long)
        )
        if self.key_dropout_embed is None:
            return embedded
        token = self.key_dropout_embed[key_index].to(dtype=embedded.dtype)
        return torch.where(drop_mask[:, None, None], token, embedded)


__all__ = [
    "MiraActionEncoder",
    "MiraActionInput",
]
