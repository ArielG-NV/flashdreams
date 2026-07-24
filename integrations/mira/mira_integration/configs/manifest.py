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

"""Loader, validation, and pipeline generation for MIRA demo manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nvtx
from omegaconf import OmegaConf

from mira_integration.configs.schema import (
    MiraInputBinding,
    MiraManifest,
    MiraModelMetadata,
    MiraWebRTCModelConfig,
)
from mira_integration.decoder import MiraDecoderConfig
from mira_integration.encoder import MiraControlEncoderConfig
from mira_integration.network import MiraDiTConfig
from mira_integration.pipeline import MiraPipelineConfig
from mira_integration.scheduler import MiraDiffusionModelConfig, MiraFlowSchedulerConfig
from mira_integration.transformer import MiraTransformerConfig


@nvtx.annotate()
def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


@nvtx.annotate()
def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


@nvtx.annotate()
def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


@nvtx.annotate()
def _browser_keys(value: Any, location: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{location} must be a non-empty list of strings")
    keys = tuple(value)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{location} must be a non-empty list of strings")
    if not keys[0].strip():
        raise ValueError(f"{location}[0] must be a canonical browser key")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{location} contains duplicate keys")
    return keys


@nvtx.annotate()
def _load_input_maps(
    root: Mapping[str, Any],
) -> dict[str, tuple[MiraInputBinding, ...]]:
    raw_maps = _mapping(root.get("input-map"), "input-map")
    input_maps: dict[str, tuple[MiraInputBinding, ...]] = {}
    for map_id, raw_inputs in raw_maps.items():
        map_name = _string(map_id, "input-map id")
        inputs = _mapping(raw_inputs, f"input-map.{map_name}")
        bindings: list[MiraInputBinding] = []
        for input_id, raw_binding in inputs.items():
            input_name = _string(input_id, f"input-map.{map_name} input id")
            location = f"input-map.{map_name}.{input_name}"
            binding = _mapping(raw_binding, location)
            browser_keys = _browser_keys(
                binding.get("browser_key"), f"{location}.browser_key"
            )
            bindings.append(
                MiraInputBinding(
                    browser_key=browser_keys[0],
                    aliases=browser_keys[1:],
                    checkpoint_key=_string(
                        binding.get("checkpoint_key"), f"{location}.checkpoint_key"
                    ),
                    label=_string(binding.get("label"), f"{location}.label"),
                    action=_string(binding.get("action"), f"{location}.action"),
                    group=_string(binding.get("group"), f"{location}.group"),
                    group_description=str(binding.get("group_description", "")),
                )
            )
        if not bindings:
            raise ValueError(f"input-map.{map_name} must contain at least one input")
        canonical_keys = [binding.browser_key for binding in bindings]
        checkpoint_keys = [binding.checkpoint_key for binding in bindings]
        if len(canonical_keys) != len(set(canonical_keys)):
            raise ValueError(f"input-map.{map_name} contains duplicate browser keys")
        if len(checkpoint_keys) != len(set(checkpoint_keys)):
            raise ValueError(f"input-map.{map_name} contains duplicate checkpoint keys")
        input_maps[map_name] = tuple(bindings)
    if not input_maps:
        raise ValueError("input-map must contain at least one map")
    return input_maps


@nvtx.annotate()
def _load_demos(
    root: Mapping[str, Any],
    input_maps: Mapping[str, tuple[MiraInputBinding, ...]],
) -> dict[str, MiraModelMetadata]:
    raw_demos = _mapping(root.get("demos"), "demos")
    demos: dict[str, MiraModelMetadata] = {}
    for demo_id, raw_demo in raw_demos.items():
        name = _string(demo_id, "demo id")
        location = f"demos.{name}"
        demo = _mapping(raw_demo, location)
        input_map_id = _string(demo.get("input-map"), f"{location}.input-map")
        try:
            input_key_map = input_maps[input_map_id]
        except KeyError as exc:
            raise ValueError(
                f"{location}.input-map references unknown map {input_map_id!r}"
            ) from exc
        demos[name] = MiraModelMetadata(
            name=name,
            display_name=_string(demo.get("display_name"), f"{location}.display_name"),
            checkpoint=_string(
                demo.get("checkpoint-hugging-face"),
                f"{location}.checkpoint-hugging-face",
            ),
            player_count=_positive_int(
                demo.get("player_count"), f"{location}.player_count"
            ),
            n_context_frames=_positive_int(
                demo.get("n_context_frames"), f"{location}.n_context_frames"
            ),
            steps=_positive_int(demo.get("steps"), f"{location}.steps"),
            latent_height=_positive_int(
                demo.get("latent_height"), f"{location}.latent_height"
            ),
            latent_width=_positive_int(
                demo.get("latent_width"), f"{location}.latent_width"
            ),
            input_key_map=input_key_map,
            video_width=_positive_int(
                demo.get("video_width"), f"{location}.video_width"
            ),
            video_height=_positive_int(
                demo.get("video_height"), f"{location}.video_height"
            ),
            frames_per_chunk=_positive_int(
                demo.get("frames_per_chunk"), f"{location}.frames_per_chunk"
            ),
        )
    if not demos:
        raise ValueError("demos must contain at least one demo")
    return demos


@nvtx.annotate()
def load_manifest(path: str | Path | None) -> MiraManifest:
    """Load and validate a MIRA YAML manifest.

    Args:
        path: Explicit YAML manifest path.

    Returns:
        Validated input maps and demo metadata.

    Raises:
        ValueError: ``path`` is missing, or the manifest is malformed.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError("A MIRA manifest path is required; pass --manifest PATH.")
    manifest_path = Path(path)
    config = OmegaConf.load(manifest_path)
    raw = OmegaConf.to_container(config, resolve=True)
    root = _mapping(raw, "manifest")
    input_maps = _load_input_maps(root)
    return MiraManifest(
        input_maps=input_maps,
        demos=_load_demos(root, input_maps),
    )


@nvtx.annotate()
def build_pipeline_config(metadata: MiraModelMetadata) -> MiraPipelineConfig:
    """Build a native pipeline config from validated demo metadata.

    Args:
        metadata: Selected manifest demo metadata.

    Returns:
        Pipeline configuration matching the demo checkpoint and player layout.
    """
    player_count = metadata.player_count
    checkpoint_keys = tuple(
        binding.checkpoint_key for binding in metadata.input_key_map
    )
    return MiraPipelineConfig(
        name=metadata.name,
        enable_sync_and_profile=False,
        diffusion_model=MiraDiffusionModelConfig(
            transformer=MiraTransformerConfig(
                network=MiraDiTConfig(
                    latent_height=metadata.latent_height * player_count,
                    latent_width=metadata.latent_width,
                    n_players=player_count,
                    num_action_keys=metadata.num_action_keys,
                ),
                compile_network=True,
                use_cuda_graph=True,
            ),
            scheduler=MiraFlowSchedulerConfig(),
            seed=0,
            context_noise=0.8,
        ),
        encoder=MiraControlEncoderConfig(checkpoint_keys=checkpoint_keys),
        decoder=MiraDecoderConfig(
            n_players=player_count,
            compile_core=True,
            causal_temporal_attention_backend="triton",
            use_cuda_graph=True,
        ),
        model_repo=metadata.checkpoint,
        n_players=player_count,
        n_context_frames=metadata.n_context_frames,
    )


@nvtx.annotate()
def load_demo_config(
    manifest_path: str | Path | None,
    demo_name: str | None,
) -> MiraWebRTCModelConfig:
    """Load one named demo and generate its native pipeline config.

    Args:
        manifest_path: Explicit YAML manifest path.
        demo_name: Key to select from the manifest's ``demos`` mapping.

    Returns:
        Selected metadata and its generated pipeline configuration.

    Raises:
        ValueError: ``demo_name`` is missing or absent from ``demos``.
    """
    manifest = load_manifest(manifest_path)
    if not isinstance(demo_name, str) or not demo_name.strip():
        raise ValueError("A MIRA demo name is required; pass --demo NAME.")
    try:
        metadata = manifest.demos[demo_name]
    except KeyError as exc:
        choices = ", ".join(sorted(manifest.demos))
        raise ValueError(
            f"Unknown MIRA demo {demo_name!r}; choose from demos: {choices}"
        ) from exc
    return MiraWebRTCModelConfig(
        metadata=metadata,
        pipeline=build_pipeline_config(metadata),
    )


load_mira_manifest = load_manifest
"""Explicit alias for callers that import multiple manifest loaders."""

__all__ = [
    "build_pipeline_config",
    "load_demo_config",
    "load_manifest",
    "load_mira_manifest",
]
