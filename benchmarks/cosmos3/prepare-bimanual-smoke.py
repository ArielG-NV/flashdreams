# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Cosmos3 benchmark support."""

import json
from pathlib import Path

import numpy as np
from cosmos3_nano_sim_bimanual.apps.handdreams.agibot_replay import (
    prepare_dataset_replays,
)
from cosmos3_nano_sim_bimanual.apps.handdreams.scenes import (
    _prepare_runtime_canvas,
    load_scene_data,
)
from PIL import Image

here = Path(__file__).resolve().parent
count = 96
out = here / "results/three-seconds/bimanual-inputs"
out.mkdir(parents=True, exist_ok=True)
scene_path, actions, domains = prepare_dataset_replays()["fold-tshirt"]
sample = load_scene_data(scene_path)
image = _prepare_runtime_canvas(
    sample["init_rgb"],
    source_content_height=sample["content_height"],
    source_content_width=sample["content_width"],
)
Image.fromarray(image).save(out / "initial.png")
actions = actions[:count]
domains = domains[:count]
assert len(actions) == count
np.save(out / "actions.npy", actions)
(out / "input.json").write_text(
    json.dumps(
        {
            "prompt": sample["prompt"],
            "fps": sample["fps"],
            "domain_id": int(domains[0]),
            "domain_ids": domains.tolist(),
            "action_rows": count,
            "canvas": [832, 480],
        },
        indent=2,
    )
)
