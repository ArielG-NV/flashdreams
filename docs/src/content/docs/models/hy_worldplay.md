---
title: 'HY-WorldPlay'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="model-link-row">
  <a class="model-link-button" href="https://3d-models.hunyuan.tencent.com/world/" target="_blank" rel="noopener noreferrer">Project page</a>
  <a class="model-link-button" href="https://github.com/Tencent-Hunyuan/HY-WorldPlay" target="_blank" rel="noopener noreferrer">Official code</a>
</div>

Introduced by [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/HY-WorldPlay), HY-WorldPlay is a
real-time interactive image-to-video (I2V) world model with action + camera-trajectory conditioning and
reconstituted-context memory. FlashDreams ships a native port of the distilled WAN-5B variant (Wan 2.2
TI2V-5B backbone, 4-step distilled Euler).

See the [HY-WorldPlay integration reference](../repository/integrations_v2/hy_worldplay/README.md) for configuration and Cam2V
application wiring.

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<p class="model-footnote">
  Generated with FlashDreams' native HY-WorldPlay WAN-5B I2V pipeline.
</p>

## Installation

```bash

# from the repo root
uv sync --package flashdreams-hy-worldplay --inexact

```

Running the model requires a CUDA-capable NVIDIA GPU. Export `HF_TOKEN` with
read access to `tencent/HY-WorldPlay` before the first checkpoint download.

## Running the method

HY-WorldPlay WAN-5B is image-to-video only. Its model package binds the
pipeline directly to the reusable Cam2V v2 application:

```bash

uv run --no-sync flashdreams-run-v2 cam2v-hy-worldplay \
    --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

:::note

This application uses `flashdreams-run-v2`.

::: 

Use `W`/`S` to move, `A`/`D` to yaw, `Q`/`E` to strafe, and
`I`/`K` to pitch. The binding converts live camera poses to HY's
latent-rate PRoPE, action, and memory inputs. Application arguments follow
`--`; inspect them with:

```bash

uv run --no-sync flashdreams-run-v2 cam2v-hy-worldplay -- --help

```

Some generated samples from the above commands:

<div class="model-video-grid">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      a person walking
    </div>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-4.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      Walking through a seaside village
    </div>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-8.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      Walking through a snowy forest
    </div>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-9.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      Walking toward a castle
    </div>
  </div>
</div>

## Profiling benchmark

Here is the profiling benchmark on total DiT + VAE-decode runtime for FlashDreams HY-WorldPlay
compared to the [official HY-WorldPlay implementation](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
under matched settings.

 <figure class="benchmark-figure-wrap">
   <div
     id="hy-worldplay-benchmark-chart"
     class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/hy_worldplay/perf-0530.json"
    data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="HY-WorldPlay benchmark chart"
   ></div>
   <figcaption>
     <p class="model-footnote">
       This chart shows total DiT + VAE-decode runtime per autoregressive chunk (4 diffusion steps) in
       milliseconds, at steady state (median of the post-warmup chunks), measured at num_chunk=8,
       704x1280, seed=0 on a single GB300. For an apples-to-apples comparison, both implementations are
       forced to use the cuDNN attention backend and torch.compile under matched runtime settings.
       For the official HY-WorldPlay implementation, see
       <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/hy_worldplay/tests/parity_check">this instruction</a>.
     </p>
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>
