# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared raw image, driving controls and caption for the Auto benchmark."""

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from cosmos3_nano_sim_auto.impl import app as app_module
from cosmos3_nano_sim_auto.impl.app import Cosmos3NanoSimAutoApplication
from cosmos3_nano_sim_auto.impl.conditioning.control_provider import RenderedControls
from flashdreams.runtime_v2.user_input_events import UserInputEvents

here = Path(__file__).resolve().parent
out = here / "results/three-seconds/auto-inputs"
out.mkdir(parents=True, exist_ok=True)
captured: dict[str, Any] = {}
original_tokens = RenderedControls.tokens
original_conditions = app_module.condition_latents
original_prompt = app_module.prompt_token_ids


def capture_conditions(clips, *args, **kwargs):
    if clips:
        captured["raw_initial"] = (
            ((clips[0][0].detach().cpu().float() + 1) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
    return original_conditions(clips, *args, **kwargs)


def capture_prompt(sample, *args, **kwargs):
    captured["camera"] = sample.cameras[0]
    captured["caption"] = (
        sample.view_captions[0] if sample.view_captions else sample.prompt
    )
    return original_prompt(sample, *args, **kwargs)


def capture_tokens(self, start, end):
    render = self._render
    if render is not None:

        def capture_render(*args, **kwargs):
            frames = render(*args, **kwargs)
            captured["raw_controls"] = frames.detach().cpu().numpy()
            return frames

        self._render = capture_render
    try:
        tokens = original_tokens(self, start, end)
    finally:
        self._render = render
    return tokens


patches = ExitStack()
patches.enter_context(patch.object(RenderedControls, "tokens", capture_tokens))
patches.enter_context(patch.object(app_module, "condition_latents", capture_conditions))
patches.enter_context(patch.object(app_module, "prompt_token_ids", capture_prompt))
app = Cosmos3NanoSimAutoApplication()
session = None
try:
    with torch.inference_mode():
        app.init(
            [
                "--views",
                "1",
                "--frames",
                "97",
                "--history-frames",
                "0",
                "--seed",
                "0",
                "--outlive-caption",
                "--compile-denoise",
                "off",
                "--checkpoint",
                "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/resolve/c3fa1378e93710242501a6288d7c9e7c9084a580/model.safetensors.index.json",
            ]
        )
        session = app.create_session(app.session_desc())
        session.init()
        session.model_loop.step(0, UserInputEvents([]))
        assert captured["raw_controls"].shape == (97, 480, 832, 3)
        assert captured["raw_initial"].shape == (480, 832, 3)
        np.savez_compressed(
            out / "raw.npz",
            initial=captured["raw_initial"],
            controls=captured["raw_controls"],
        )
        (out / "raw.json").write_text(
            json.dumps(
                {"camera": captured["camera"], "caption": captured["caption"]}, indent=2
            )
        )

finally:
    with torch.inference_mode():
        if session:
            session.close()
        app.close()
    patches.close()
