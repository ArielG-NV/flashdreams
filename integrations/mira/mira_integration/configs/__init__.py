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

"""Explicit manifest loading and MIRA demo dispatch configuration."""

import nvtx

from mira_integration.configs.manifest import (
    build_pipeline_config,
    load_demo_config,
    load_manifest,
    load_mira_manifest,
)
from mira_integration.configs.schema import (
    MiraInputBinding,
    MiraManifest,
    MiraModelMetadata,
    MiraWebRTCModelConfig,
    preview_grid_dimensions,
)
from mira_integration.runner import MiraDemoRunnerConfig

RUNNER_MIRA = MiraDemoRunnerConfig(
    runner_name="mira",
    description="Run a named MIRA demo from an explicit YAML manifest.",
)
"""Manifest-selecting ``flashdreams-run mira`` dispatcher."""

_NVTX_ANNOTATE = nvtx.annotate

__all__ = [
    "MiraInputBinding",
    "MiraManifest",
    "MiraModelMetadata",
    "MiraWebRTCModelConfig",
    "build_pipeline_config",
    "load_demo_config",
    "load_manifest",
    "load_mira_manifest",
    "preview_grid_dimensions",
]
