---
title: 'FlashDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams is an inference and serving runtime for turning
autoregressive video and world models into live, controllable
simulations. It runs the model in a continuous loop, carrying
state forward and streaming frames while new actions or sensor
inputs change what happens next, whether the application is a
game world, an autonomous-vehicle simulator, robotic policy
testing, or a virtual training environment.

[Get Started!](quickstart/index.md)
[GitHub](https://github.com/NVIDIA/flashdreams)
[Contribute](community/index.md)

<img alt="FlashDreams quick intro animation" src="_static/promo/flashdreams-promo.avif" />

## Why FlashDreams?

A world model learns to generate and evolve an environment over time. In
practice that usually means video, but the same idea extends to actions,
state, audio, sensor input, and control signals. Serving one means keeping
a session alive while input, model state, GPU inference, and output advance
together, rather than producing a single static clip, which is what makes
interactive simulation, robotics, autonomy, and game-like experiences
possible.

<img alt="Offline one-shot video inference compared with online autoregressive world-model serving." src="_static/diagrams/compare-offline-online-video-model-v2.jpg" />

FlashDreams is built for that real-time case: a closed-loop world-model
demo, a driving simulator, an interactive scene rollout. Generating
high-quality video is not enough on its own. The runtime has to keep an
interactive session responsive while the model continues to advance the
world. That comes down to four things:

### Low latency

Keep the interaction responsive when controls, sensors, or user
input change.

### High throughput

Keep the GPU busy across autoregressive steps and multi-GPU
execution.

### Steady streaming generation

Stream frames or chunks at a steady pace while the session
continues.

### World-state evolution

Carry rolling state forward so the generated world evolves across
steps.

## Performance

Each tile shows the speedup over a separate existing implementation of
the same model. Both runs use the same weights on the same GPU, so the
gain comes from FlashDreams' runtime alone. Each tile links to the
profiling chart on its model page.

### [Self-Forcing: 2.12× speedup](models/self_forcing.md#profiling-benchmark)

### [LingBot-World: 3.10× speedup](models/lingbot_world.md#profiling-benchmark)

### [Wan2.1: 1.40× speedup](models/wan21.md#profiling-benchmark)

### [FlashVSR: 1.42× speedup](models/flashvsr.md#profiling-benchmark)

## Try FlashDreams!

FlashDreams brings best-in-class per-step latency to interactive
autoregressive video and world models: multiple integrated models across
streaming and bidirectional methods, multi-GPU execution, and CLI entry
points for runner presets, demos, and v2 applications.

The [Get Started guide](quickstart/index.md) walks from a fresh
checkout to running OmniDreams, an interactive driving world-model demo
built on FlashDreams.

## Supported Models

Streaming and autoregressive model implementations emit per-step output with
sub-second latency once warm; bidirectional model implementations are kept as
full-block parity references. Each model page carries the canonical
invocation, the checkpoint source, and the per-implementation knobs.

### [OmniDreams](models/omnidreams.md)

Interactive world simulator for autonomous vehicles.

### [Self-Forcing](models/self_forcing.md)

Autoregressive text-to-video based on Wan 2.1.

### [Causal-Forcing](models/causal_forcing.md)

Autoregressive text/image-to-video based on Wan 2.1.

### [Causal Wan 2.2](models/causal_wan22.md)

Autoregressive text-to-video based on Wan 2.2 from FastVideo.

### [LingBot-World](models/lingbot_world.md)

Camera-controllable image-to-video world model.

### [Waypoint 1.5](models/waypoint.md)

Interactive image-established world model controlled by keyboard and mouse.

### [SANA-WM](models/sana_wm_streaming.md)

Camera-controlled world model with streaming and bidirectional variants.

### [FlashVSR](models/flashvsr.md)

Streaming video super-resolution.

### [SwiftVR](models/swiftvr.md)

Real-time one-step streaming video restoration with 2x and 4x presets.

### [Wan 2.1 (bidirectional)](models/wan21.md)

Bidirectional video generation model that supports both
text-to-video and image-to-video.

### [Wan 2.2 TI2V-5B (bidirectional)](models/wan22.md)

Bidirectional text-and-image-to-video generation in one full-clip rollout.

### [Cosmos-Predict2.5 (bidirectional)](models/cosmos_predict2.md)

Bidirectional Cosmos-Predict2 reference implementations (T2V / I2V, 2B).

   here = order in the navbar.
