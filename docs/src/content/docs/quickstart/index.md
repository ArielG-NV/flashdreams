---
title: 'Get Started'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Welcome to FlashDreams! This page will guide you from a fresh checkout to a running model.
This guide will use [NVIDIA OmniDreams](../models/omnidreams.md), a interactive driving world model, as the
example. Refer to our [model gallery](../models/index.md) for a guide on running other models.

## Install

First, clone FlashDreams:
```bash
git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
```

Now we need to install `uv`, FlashDreams's package manager.
[Installation instructions here](https://docs.astral.sh/uv/getting-started/installation/).

With `uv` installed, clone the repository and synchronize `uv` to the OmniDreams workspace via:
```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
```

Most runs need a Hugging Face token for model/example-asset downloads. For OmniDreams, use a token with
read access to
[nvidia/omni-dreams-models](https://huggingface.co/nvidia/omni-dreams-models) and
[nvidia/omni-dreams-scenes](https://huggingface.co/datasets/nvidia/omni-dreams-scenes):

```bash

export HF_TOKEN=<your-hf-token>

```

## Run your first model

Launch the OmniDreams interactive driving demo. It runs the world model
and streams the generated camera view to a browser over WebRTC:

```bash

uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams --mode webrtc \
    --host 0.0.0.0 --port 8089
```

Then open `http://<server-ip>:8089/` in a browser on the same network (use
`localhost` on the same machine). The first launch spends several
minutes loading checkpoints and compiling kernels; later launches reuse
the cached assets.

Inspect the application arguments without loading checkpoints:

```bash

uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams -- --help

```

See [/models/omnidreams](../models/omnidreams.md) for other options & configurations to run OmniDreams.

## Where to next

- [/models/index](../models/index.md): every FlashDreams default-shipped model with CLI-slug to run it.
- [CLI and API Reference](../api/index.md): exact commands and Python
  contracts for the inference API, demo API, and separate v2 application
  stack.
- [/developer_guides/inference_pipeline_overview](../developer_guides/inference_pipeline_overview.md): the generation
  loop end to end: KV cache, ring attention, CUDA-graph capture.
- [/developer_guides/config_system](../developer_guides/config_system.md): the configuration layer
  every method shares.
- [/developer_guides/new_integration](../developer_guides/new_integration.md): adding a new model or
  method as a plugin.
- [/troubleshooting](../troubleshooting.md): common first-run failures and fixes.
