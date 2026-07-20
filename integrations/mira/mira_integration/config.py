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

"""Static native MIRA pipeline and runner configs."""

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import RunnerConfig

from mira_integration.decoder import MiraDecoderConfig
from mira_integration.encoder import MiraControlEncoderConfig
from mira_integration.network import MiraDiTConfig
from mira_integration.pipeline import MiraPipelineConfig
from mira_integration.runner import MiraDemoRunnerConfig
from mira_integration.scheduler import MiraDiffusionModelConfig, MiraFlowSchedulerConfig
from mira_integration.transformer import MiraTransformerConfig

PIPELINE_MIRA_MINI_1B = MiraPipelineConfig(
    name="mira-mini-1b-demo",
    enable_sync_and_profile=True,
    diffusion_model=MiraDiffusionModelConfig(
        transformer=MiraTransformerConfig(network=MiraDiTConfig()),
        scheduler=MiraFlowSchedulerConfig(),
        seed=0,
        context_noise=0.8,
    ),
    encoder=MiraControlEncoderConfig(),
    decoder=MiraDecoderConfig(),
)
"""Published MIRA Mini 1B executed by FlashDreams-native components."""

MIRA_CONFIGS: dict[str, StreamInferencePipelineConfig] = {
    PIPELINE_MIRA_MINI_1B.name: PIPELINE_MIRA_MINI_1B
}

RUNNER_MIRA_MINI_1B_DEMO = MiraDemoRunnerConfig(
    runner_name=PIPELINE_MIRA_MINI_1B.name,
    description=("MIRA Mini 1B native FlashDreams car-soccer world-model demo."),
    pipeline=PIPELINE_MIRA_MINI_1B,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    RUNNER_MIRA_MINI_1B_DEMO.runner_name: RUNNER_MIRA_MINI_1B_DEMO
}

__all__ = [
    "MIRA_CONFIGS",
    "PIPELINE_MIRA_MINI_1B",
    "RUNNER_CONFIGS",
    "RUNNER_MIRA_MINI_1B_DEMO",
]
