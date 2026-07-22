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

"""MIRA flow-integration scheduler implemented through FlashDreams contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import nvtx
import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowPredictor,
    Scheduler,
    SchedulerConfig,
)
from flashdreams.infra.diffusion.model import DiffusionModelConfig


@dataclass(kw_only=True)
class MiraDiffusionModelConfig(DiffusionModelConfig):
    """Diffusion-model config with MIRA's continuous cache-update time."""

    context_noise: float = 0.8
    """Flow time used to re-noise the generated latent for KV finalization."""


@dataclass(kw_only=True)
class MiraFlowSchedulerConfig(SchedulerConfig):
    """Config for MIRA's increasing-time Euler flow sampler."""

    _target: type["MiraFlowScheduler"] = field(
        default_factory=lambda: MiraFlowScheduler
    )

    num_inference_steps: int = 2
    """Euler integration step count."""

    schedule_type: Literal["linear", "linear_quadratic"] = "linear_quadratic"
    """Spacing of integration points from noise ``tau=0`` to clean ``tau=1``."""


class MiraFlowScheduler(Scheduler):
    """Integrate MIRA's predicted velocity from Gaussian noise to clean latent."""

    config: MiraFlowSchedulerConfig

    def __init__(self, config: MiraFlowSchedulerConfig) -> None:
        super().__init__(config)
        self.config = config

    @nvtx.annotate("MiraFlowScheduler.set_num_inference_steps")
    def set_num_inference_steps(self, value: int) -> None:
        """Set the rollout-specific Euler step count."""
        if value < 1:
            raise ValueError("num_inference_steps must be positive")
        self.config.num_inference_steps = value

    @nvtx.annotate("MiraFlowScheduler._schedule")
    def _schedule(self, device: torch.device) -> Tensor:
        steps = self.config.num_inference_steps
        if self.config.schedule_type == "linear":
            return torch.linspace(0.0, 1.0, steps + 1, device=device)
        if steps < 2:
            return torch.tensor((0.0, 1.0), device=device)
        linear_steps = steps // 2
        linear = torch.linspace(0.0, 0.1, linear_steps + 1, device=device)
        quadratic_steps = steps - linear_steps
        quadratic = torch.linspace(
            torch.sqrt(linear[-1]), 1.0, quadratic_steps + 1, device=device
        ).square()
        return torch.cat((linear[:-1], quadratic))

    @nvtx.annotate("MiraFlowScheduler.sample")
    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Run explicit Euler integration over MIRA's increasing ``tau`` grid."""
        _ = rng
        sample = initial_noise
        schedule = self._schedule(initial_noise.device)
        for tau, delta in zip(schedule[:-1], schedule.diff()):
            with nvtx.annotate("MiraFlowScheduler.sample.step"):
                timestep = tau.to(dtype=initial_noise.dtype)
                sample = sample + delta * predict_flow(sample, timestep)
        return sample.to(dtype=initial_noise.dtype)

    @nvtx.annotate("MiraFlowScheduler.add_noise")
    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Interpolate clean latent and Gaussian noise at flow time ``tau``."""
        noise = torch.empty_like(clean_input).normal_(generator=rng)
        tau = timestep.to(dtype=clean_input.dtype)
        return tau * clean_input + (1.0 - tau) * noise
