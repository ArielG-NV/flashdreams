---
title: 'Models'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams runs a growing family of world and video models (text-to-video,
image-to-video, camera-controlled, and super-resolution). Model cards identify
the supported launch path. See the [API overview](../api/index.md) for the
runtime families.

## Available models

The models come in three flavors. Streaming and autoregressive generation
methods build a video step by step and stay fast once warmed up, aiming for
sub-second latency per step; bidirectional methods produce a clip in a single
pass and serve as the quality reference for their streaming counterparts; and
super-resolution methods upscale existing frames in chunks, so their latency
scales with output resolution rather than step count. Each card identifies its
supported public launch path or programmatic pipeline access.

Streaming and autoregressive generation

### [OmniDreams](omnidreams.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.mp4" type="video/mp4">
</video>

Interactive world simulator for autonomous vehicles.

### [Self-Forcing](self_forcing.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_1.mp4" type="video/mp4">
</video>

Autoregressive text-to-video based on Wan 2.1.

### [Causal-Forcing](causal_forcing.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.mp4" type="video/mp4">
</video>

Autoregressive text/image-to-video based on Wan 2.1.

### [Causal Wan 2.2](causal_wan22.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_1.mp4" type="video/mp4">
</video>

Autoregressive text-to-video based on Wan 2.2 from FastVideo.

### [LingBot-World](lingbot_world.md)

<div class="fd-card-video-wrap">
  <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-01.mp4" type="video/mp4">
  </video>
  <video class="fd-card-video-pip" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-traj-01.mp4" type="video/mp4">
  </video>
</div>

Camera-controllable image-to-video world model.

### [Waypoint 1.5](waypoint.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://huggingface.co/Overworld/Waypoint-1.5-1B/resolve/main/assets/wp_1.5.mp4" type="video/mp4">
</video>

Interactive image-established world model controlled by keyboard and mouse.

### [HY-WorldPlay](hy_worldplay.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-hero.mp4" type="video/mp4">
</video>

Action- and camera-controllable image-to-video world model.

### [SANA-WM_streaming](sana_wm_streaming.md)

<img alt="SANA-WM streaming FlashDreams sample clip." src="../_static/model_clips/sana_wm/sana-wm-streaming.avif" />
Chunk-causal camera-controlled world model with streaming Stage-1,
refiner, and VAE paths.

Bidirectional Video Generation

### [Wan 2.1](wan21.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/wan21/wan21-t2v-1.3b-480p.mp4" type="video/mp4">
</video>

Bidirectional video generation model that supports both
text-to-video and image-to-video.

### [Cosmos-Predict2.5](cosmos_predict2.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/cosmos_predict2/cosmos2-t2v-2b-720p.mp4" type="video/mp4">
</video>

Bidirectional Cosmos-Predict2 reference implementations (T2V / I2V, 2B).

### [SANA-WM_bidirectional](sana_wm_bidirectional.md)

<img alt="SANA-WM bidirectional FlashDreams sample clip." src="../_static/model_clips/sana_wm/sana-wm-bidirectional.avif" />
Bidirectional camera-controlled world model (Stage-1 DiT + LTX-2
refiner, 2.6B).

Super-resolution

### [FlashVSR](flashvsr.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.mp4" type="video/mp4">
</video>

Streaming video super-resolution.

## Running a model yourself

Follow the selected model card for its supported command. Current public model
applications use `flashdreams-run-v2 <APPLICATION_SLUG>`; the
[CLI reference](../api/cli.md) explains runtime modes and the `--` separator.
The [v2 integration reference](../repository/integrations_v2/README.md) lists
every registered package, including Wan 2.2 and SwiftVR integrations that do
not yet have showcase model cards.

### Adding your own model

See [/developer_guides/new_integration](../developer_guides/new_integration.md) for model integration and registration
guidance.

## Related

- Follow the [/quickstart/index](../quickstart/index.md) for the shortest path to
  running these methods on your own hardware.
- The [/developer_guides/index](../developer_guides/index.md) cover the architecture behind the
  methods you can run today.
- [/community/index](../community/index.md) lists the channels to use if a method on
  this page does not run on your hardware.
- Browse the source on GitHub at [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams).
