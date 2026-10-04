---
title: 'Get Started'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Welcome to FlashDreams! This page will guide you from a fresh checkout
of the repository to a running model. It uses [NVIDIA OmniDreams](../models/omnidreams.md), the interactive driving world model, as the
example; the [model gallery](../models/index.md) lists the run command
for every other model.

## Install

FlashDreams uses the `uv` Python package manager (`installation
instructions <https://docs.astral.sh/uv/getting-started/installation/>`_).
With `uv` installed, clone the repository and synchronize the
OmniDreams workspace:

```bash

git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact

```

Most runs need a Hugging Face token. For OmniDreams, use a token with
read access to
[nvidia/omni-dreams-models](https://huggingface.co/nvidia/omni-dreams-models) and
[nvidia/omni-dreams-scenes](https://huggingface.co/datasets/nvidia/omni-dreams-scenes):

```bash

export HF_TOKEN=<your-hf-token>

```

For development containers, see [Docker](../repository/docker/README.md).
For common first-run failures, see [/troubleshooting](../troubleshooting.md).

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

See [/models/omnidreams](../models/omnidreams.md) for scripted generation, scene variants,
native-window serving, and multi-GPU options.

This example uses the v2 application stack. See the [API overview](../api/index.md) before integrating a model or demo programmatically.

## Where to next

- [/models/index](../models/index.md): every shipped model with its CLI slug and the
  command to run it.
- [/models/omnidreams](../models/omnidreams.md): drive a world model in real time with the
  `native-window` or `webrtc` presentation mode.
- [/developer_guides/inference_pipeline_overview](../developer_guides/inference_pipeline_overview.md): the generation
  loop end to end: KV cache, ring attention, CUDA-graph capture.
- [/developer_guides/config_system](../developer_guides/config_system.md): the configuration layer
  every method shares.
- [/developer_guides/new_integration](../developer_guides/new_integration.md): adding a new model or
  method as a plugin.
- [CLI and API Reference](../api/index.md): exact commands and Python
  contracts for the inference API, demo API, and separate v2 application
  stack.
- [/troubleshooting](../troubleshooting.md): common first-run failures and fixes.
