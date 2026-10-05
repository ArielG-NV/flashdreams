---
title: 'LingBot-World'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

[Project page](https://technology.robbyant.com/lingbot-world)
[Official code](https://github.com/robbyant/lingbot-world)

Introduced by [Robbyant](https://technology.robbyant.com/), LingBot-World is a camera-controllable image-to-video
(I2V) world model with streaming inference and context-parallel runtime support. This page covers both the original
[LingBot-World v1](https://github.com/robbyant/lingbot-world) and the newer 14B causal-fast
[LingBot-World v2](https://github.com/Robbyant/lingbot-world-v2) checkpoints.

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://gw.alipayobjects.com/v/huamei_u94ywh/afts/video/XQk7Rb44qJwAAAAAgfAAAAgAfoeUAQBr" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<p class="model-footnote">
  Teaser video source:
  <a href="https://technology.robbyant.com/lingbot-world">LingBot-World project page</a>.
</p>

## Run with FlashDreams

From the repository root:

```bash
uv sync --package flashdreams-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

## Developer details

[Integration source](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/lingbot) · [Pipeline configuration](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/lingbot/config.py) · [Application guide](../repository/integrations_v2/lingbot/apps/cam2v/README.md) · [Tests](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/lingbot/tests)

### Configurations and behavior

Sample data is downloaded from the
[LingBot-World v2 repository](https://github.com/Robbyant/lingbot-world-v2/tree/main/examples).
Valid `--example-idx` values are `0` through `5`; examples `0`, `1`,
`2`, and `5` also provide an upstream `prompt.txt`. Note the single GPU command might run
out of memory for large `--total-blocks` values.

The package exposes the following pipeline configurations for programmatic
use:

| Method | Description |
| --- | --- |
| `lingbot-world-fast` | Official camera-control I2V (Wan VAE decoder, full KV-cache). |
| `lingbot-world-fast-taehv-window15-sink3` | Efficient streaming configuration: TAEHV decoder, `window_size_t=15` + `sink_size_t=3` streaming KV-cache. |
| `lingbot-world-v2-14b-causal-fast` | LingBot-World V2 14B causal-fast on the shared LingBot pipeline (Wan VAE decoder, 4-step). See [lingbot-world-v2](lingbot_world.md#lingbot-world-v2). |
| `lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` | LingBot-World V2 14B causal-fast with the TAEHV decoder, `window_size_t=15` + `sink_size_t=3` streaming KV-cache. |

<a id="lingbot-world-v2"></a>

### Requirements

- **Minimum VRAM**: ~120 GB.
- **PyTorch**: >= 2.9.

### LingBot-World V2

LingBot-World V2 is the newer 14B causal-fast checkpoint from Robbyant. It
uses the same architecture and pipeline code as v1; only the checkpoint
configuration changes. See the canonical repository at
[Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2).

Two V2 pipeline configs are exported:

| Method | Description |
| --- | --- |
| `lingbot-world-v2-14b-causal-fast` | LingBot-World V2 14B causal-fast on the shared LingBot pipeline (Wan VAE decoder, full KV-cache). |
| `lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` | V2 checkpoint with the efficient streaming preset: TAEHV decoder, `window_size_t=15` + `sink_size_t=3` streaming KV-cache. |

The V2 checkpoint (~70 GB) is pulled from
`huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast` on first run.
Export `HF_TOKEN` first.

### What to expect

- **Example data**: `--example-data` downloads `image.jpg`,
  `intrinsics.npy`, and `poses.npy` from the
  [canonical examples folder](https://github.com/Robbyant/lingbot-world-v2/tree/main/examples)
  into the FlashDreams user cache under `example_data/lingbot_world/<NN>/`
  (`<NN>` matches `--example-idx`). For examples `0`, `1`, `2`, and
  `5`, it also downloads `prompt.txt`. Cached after first run; no
  credentials needed.
- **Model checkpoint**: ~70 GB pulled from
  `huggingface.co/robbyant/lingbot-world-fast` on first run, cached
  under `$HF_HOME`. Export `HF_TOKEN` first.
- **Disk**: keep ~200 GB free for the model + HF cache. Hosts under
  \~100 GB have been seen to run out mid-load.
- **First launch**: a few minutes (download + Triton autotuning +
  CUDA-graph warmup). Subsequent launches reuse the caches.
- **Outputs**: select MP4, WebRTC, or native-window presentation with the
  `flashdreams-run-v2` runtime arguments. The Cam2V defaults are 16 FPS and
  464×832.

See [/developer_guides/inference_pipeline_overview](../developer_guides/inference_pipeline_overview.md) for what one
autoregressive chunk does end-to-end.

Some generated samples from the above commands:

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-01.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <video autoplay muted loop playsinline preload="metadata" style="position: absolute; right: 10px; bottom: 10px; width: 33.3333%; opacity: 0.7; border-radius: 8px; pointer-events: none;">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-traj-01.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      example_idx: 01
    </div>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-02.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <video autoplay muted loop playsinline preload="metadata" style="position: absolute; right: 10px; bottom: 10px; width: 33.3333%; opacity: 0.7; border-radius: 8px; pointer-events: none;">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-traj-02.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      example_idx: 02
    </div>
  </div>
</div>

### Launch the interactive server

Run the same Cam2V application in WebRTC mode:

```bash

uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

When successfully connected, the browser-based UI looks like this:

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-webrtc-recording-0529.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

### Profiling benchmark

Here is the profiling benchmark on total DiT runtime for FlashDreams LingBot-World
compared to the [official LingBot-World implementation](https://github.com/robbyant/lingbot-world)
and [LightX2V](https://github.com/ModelTC/lightx2v) under
matched settings.

 <figure class="benchmark-figure-wrap">
   <div
     id="lingbot-world-benchmark-chart"
     class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/lingbot_world/perf-0521.json"
    data-benchmark-series="official:Official Impl:#3b82f6;lightx2v:LightX2V:#f59e0b;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="LingBot-World benchmark chart"
   ></div>
   <figcaption>
     <p class="model-footnote">
       This chart shows total DiT runtime (4 diffusion steps) in milliseconds at the 6th autoregressive rollout on 4x GPUs.
       For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings,
       and all runs use Ulysses sequence parallelism for multi-GPU inference.
       The historical comparison used the upstream official implementation
       and LightX2V as the two external baselines.
     </p>
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

### Citation

If you use LingBot-World, please cite the original work:

```bibtex

@article{lingbot-world,
      title={Advancing Open-source World Models},
      author={Robbyant Team and Zelin Gao and Qiuyu Wang and Yanhong Zeng and Jiapeng Zhu and Ka Leong Cheng and Yixuan Li and Hanlin Wang and Yinghao Xu and Shuailei Ma and Yihang Chen and Jie Liu and Yansong Cheng and Yao Yao and Jiayi Zhu and Yihao Meng and Kecheng Zheng and Qingyan Bai and Jingye Chen and Zehong Shen and Yue Yu and Xing Zhu and Yujun Shen and Hao Ouyang},
      journal={arXiv preprint arXiv:2601.20540},
      year={2026}
}
```
