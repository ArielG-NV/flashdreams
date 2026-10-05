---
title: 'Self-Forcing'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

[Project page](https://self-forcing.github.io/)
[arXiv paper](https://arxiv.org/abs/2506.08009)
[Official code](https://github.com/guandeh17/Self-Forcing)

Self-Forcing is a text-to-video (T2V) model based on [Wan2.1](wan21.md).
It uses a training paradigm for autoregressive video diffusion that simulates
inference-time rollout during training with KV caching, reducing the train-test
gap and enabling efficient streaming generation quality.

![Self-Forcing teaser figure.](https://self-forcing.github.io/static/teaser.jpg)

<p class="model-footnote">
  Teaser image source:
  <a href="https://self-forcing.github.io/">Self-Forcing project page</a>.
</p>

## Run with FlashDreams

From the repository root:

```bash
uv sync --package flashdreams-self-forcing --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b \
  --output-path artifacts/self-forcing.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7
```

## Developer details

[Integration source](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing) · [Pipeline configuration](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/self_forcing/config.py) · [Application guide](../repository/integrations_v2/self_forcing/apps/t2v/README.md) · [Tests](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing/tests)

### Configurations and behavior

Available configurations:

| Method | Description |
| --- | --- |
| `self-forcing-wan2.1-t2v-1.3b` | Official checkpoint. |
| `self-forcing-wan2.1-t2v-1.3b-taehv` | Official checkpoint. Swap Wan VAE decoder with the faster TAEHV decoder. |
| `self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` | Steady long-rollout preset: static sink=5 + rolling window=7, with KVCache-relative RoPE. |

### Requirements

- **Minimum VRAM**: ~24 GB.
- **PyTorch**: >= 2.9.

### What to expect

- **Prompt**: `--prompt` is required.
- **Total blocks**: `--total-blocks N` runs `N` autoregressive
  chunks. Commands here use `7` for a fast demo; the config default
  is `60` for full rollouts. See
  [/developer_guides/inference_pipeline_overview](../developer_guides/inference_pipeline_overview.md) for what one
  chunk does end-to-end.
- **Outputs**: `--output-path` selects the MP4 destination. The application
  emits 16 FPS at 832×480.

Measured runtimes on H100 80GB with `--total-blocks 7`:

| Setup | First run (cold) | Subsequent runs |
| --- | --- | --- |
| 1× H100 PCIe | ~6.9 min | ~42 s |
| 4× H100 HBM3 (`torchrun --nproc_per_node=4`) | ~8.6 min | ~73 s |

Cold runs are dominated by the first two AR blocks (Triton autotuning +
CUDA-graph warmup); steady-state blocks are sub-second.

Per-block steady-state on 4 GPUs is ~2× faster (~251 ms vs ~500 ms),
but per-rank autotune + NCCL overhead makes 4 GPUs end-to-end slower
than 1 GPU at `--total-blocks 7`. Multi-GPU pays off once steady-state
dominates warmup — use it for `--total-blocks 60+`.

Some generated samples from the above commands:

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <!-- <div class="model-video-placeholder">Video placeholder</div> -->
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      A close-up shot of a ceramic teacup slowly pouring water into a glass mug. The water flows smoothly from the spout of the teacup into the mug, creating gentle ripples as it fills up. Both cups have detailed textures, with the teacup having a matte finish and the glass mug showcasing clear transparency. The background is a blurred kitchen countertop, adding context without distracting from the central action. The pouring motion is fluid and natural, emphasizing the interaction between the two cups.
    </div>
  </div>
  <div class="model-video-card">
    <!-- <div class="model-video-placeholder">Video placeholder</div> -->
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_6.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      A dramatic and dynamic scene in the style of a disaster movie, depicting a powerful tsunami rushing through a narrow alley in Bulgaria. The water is turbulent and chaotic, with waves crashing violently against the walls and buildings on either side. The alley is lined with old, weathered houses, their facades partially submerged and splintered. The camera angle is low, capturing the full force of the tsunami as it surges forward, creating a sense of urgency and danger. People can be seen running frantically, adding to the chaos. The background features a distant horizon, hinting at the larger scale of the tsunami. A dynamic, sweeping shot from a low-angle perspective, emphasizing the movement and intensity of the event.
    </div>
  </div>
</div>

### Profiling benchmark

Here is the profiling benchmark on total DiT runtime for FlashDreams Self-Forcing compared to
the [official Self-Forcing implementation](https://github.com/guandeh17/Self-Forcing)
and the [FastVideo implementation](https://github.com/hao-ai-lab/FastVideo)
under matched settings.

 <figure class="benchmark-figure-wrap">
   <div
     id="self-forcing-benchmark-chart"
     class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/self_forcing/perf-0521.json"
     data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="Self-Forcing benchmark chart"
   ></div>
   <figcaption>
    <p class="model-footnote">
       This chart shows the DiT total runtime (4 denoising steps in milliseconds) at the 6th autoregressive rollout on a single GPU.
       For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend and <code>torch.compile</code> for DiT network.
       For profiling the official implementation, see
       <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing/tests/parity_check/README.md">this instruction</a>.
       For profiling the FastVideo implementation, see
       <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing/tests/baseline_fastvideo/README.md">this instruction</a>.
     </p>
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

### Citation

If you use Self-Forcing, please cite the original work:

```bibtex

@article{huang2025self,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={167283--167308},
  year={2025}
}
```
