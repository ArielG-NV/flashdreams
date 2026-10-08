---
title: 'flashdreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="flashdreams-pypireadme--flashdreams"></a>

An inference and serving runtime for autoregressive video and world models,
with a plugin architecture for model integrations.

<a id="flashdreams-pypireadme--features"></a>

## Features

- **Streaming Inference** -- Autoregressive chunk-wise video generation with
  per-rollout cache state for bounded VRAM and arbitrarily long rollouts
- **Plugin Architecture** -- Entry-point-based model discovery; third-party
  packages register runner configs that appear automatically in the CLI
- **Multi-GPU** -- Context parallelism via torchrun with automatic sharding
  across ranks
- **Performance** -- torch.compile support with CUDA graph capture and replay
- **Serving** -- WebRTC integration for real-time interactive
  applications
- **Demo runtime** -- Shared input/output modes, session drivers, warmup,
  replay, benchmarking, and demo validation above model inference

<a id="flashdreams-pypireadme--supported-models"></a>

## Supported Models

First-party integrations include Wan 2.1/2.2, Cosmos-Predict2.5,
OmniDreams, LingBot-World, and more. See the
[model gallery](../../models/index.md)
for current integrations, requirements, and launch commands.

<a id="flashdreams-pypireadme--installation"></a>

## Installation

```bash

pip install flashdreams

```

FlashDreams requires Python 3.10 or newer. Model integrations can have
additional dependencies and GPU requirements; follow the model-specific setup
in the documentation.

<a id="flashdreams-pypireadme--python-apis"></a>

## Python APIs

See the [Documentation overview](../../documentation/index.md)
for inference, demo, and v2 application interfaces.

<a id="flashdreams-pypireadme--documentation"></a>

## Documentation

- [Documentation](https://nvidia.github.io/flashdreams/main/)
- [Source and issue tracker](https://github.com/NVIDIA/flashdreams)
