<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# flashdreams

A high-performance streaming inference framework for video diffusion models
with a plugin architecture for model backends.

## Features

- **Streaming Inference** -- Autoregressive chunk-wise video generation with
  per-rollout cache state for bounded VRAM and arbitrarily long rollouts
- **Plugin Architecture** -- Entry-point-based model discovery; third-party
  packages register runner configs that appear automatically in the CLI
- **Multi-GPU** -- Context parallelism via torchrun with automatic sharding
  across ranks
- **Performance** -- torch.compile support with CUDA graph capture and replay
- **Serving** -- WebRTC and gRPC integrations for real-time interactive
  applications

## Supported Models

First-party integrations: Self-Forcing, OmniDreams, LingBot-World, Wan2.1,
Causal-Forcing, FastVideo CausalWan 2.2, FlashVSR, and Cosmos-Predict2.5. See
the project page for runner slugs and per-model launch commands.

## Installation

```bash
pip install flashdreams
```

## Documentation

<https://github.com/NVIDIA/flashdreams>
