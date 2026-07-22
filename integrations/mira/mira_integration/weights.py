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

"""Native MIRA bundle resolution and strict checkpoint prefix selection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import nvtx
from huggingface_hub import snapshot_download
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint

if TYPE_CHECKING:
    from mira_integration.decoder import MiraVideoDecoder
    from mira_integration.encoder import MiraBootstrapEncoder
    from mira_integration.transformer import MiraTransformer


@nvtx.annotate("mira.weights.resolve_bundle")
def resolve_bundle(model_repo: str, local_bundle: Path | None = None) -> Path:
    """Resolve the published Hugging Face bundle or validate a local override."""
    bundle = (
        local_bundle.expanduser()
        if local_bundle is not None
        else Path(snapshot_download(model_repo))
    )
    required = (
        bundle / "world_model_config.yaml",
        bundle / "context" / "default.npz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete MIRA bundle; missing: {missing}")
    return bundle


@nvtx.annotate("mira.weights.find_world_checkpoint")
def find_world_checkpoint(bundle: Path) -> Path:
    """Return the newest published world-model checkpoint in ``bundle``."""
    checkpoints = sorted(bundle.glob("checkpoint-*/checkpoint.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No MIRA checkpoint found under {bundle}")
    return checkpoints[-1]


@nvtx.annotate("mira.weights._select")
def _select(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Strip ``prefix`` from matching state-dict keys."""
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }


@nvtx.annotate("mira.weights.load_native_weights")
def load_native_weights(
    checkpoint_path: Path,
    *,
    transformer: "MiraTransformer",
    bootstrap_encoder: "MiraBootstrapEncoder",
    decoder: "MiraVideoDecoder",
) -> None:
    """Load one checkpoint into FlashDreams-native MIRA components strictly."""
    with nvtx.annotate("mira.weights.load_native_weights.read_checkpoint"):
        checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"MIRA checkpoint has no state_dict: {checkpoint_path}")
    state = cast(dict[str, Tensor], checkpoint["state_dict"])

    with nvtx.annotate("mira.weights.load_native_weights.select_network"):
        multiplayer = any(key.startswith("single_world_model.") for key in state)
        network_state: dict[str, Tensor] = {}
        for key, value in state.items():
            native_key = key.removeprefix("single_world_model.") if multiplayer else key
            if (
                native_key == "bos"
                or native_key.startswith("action_encoder.")
                or native_key.startswith("world_model.")
                or key == "player_embedding"
                or key.startswith("player_action_projection.")
            ):
                network_state[key if key.startswith("player_") else native_key] = value
    with nvtx.annotate("mira.weights.load_native_weights.load_network"):
        transformer.network.load_state_dict(network_state, strict=True)
    codec_prefix = "single_world_model.codec." if multiplayer else "codec."
    with nvtx.annotate("mira.weights.load_native_weights.load_bootstrap_encoder"):
        bootstrap_encoder.load_state_dict(
            _select(state, f"{codec_prefix}encoder."), strict=True
        )
    with nvtx.annotate("mira.weights.load_native_weights.load_decoder"):
        decoder.load_state_dict(_select(state, f"{codec_prefix}decoder."), strict=True)
