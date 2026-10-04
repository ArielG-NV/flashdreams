---
title: 'Wan2.1'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

See the [Wan 2.1 integration reference](../repository/integrations_v2/wan21/README.md) for registered and programmatic
configurations.

[Project page](https://wan.video/)
[arXiv paper](https://arxiv.org/abs/2503.20314)
[Official code](https://github.com/Wan-Video/Wan2.1)

Wan2.1 is a bidirectional video generation model, supporting both
text-to-video (T2V) and image-to-video (I2V) tasks.

## Requirements

- **Minimum VRAM**: ~46 GB.
- **PyTorch**: >= 2.9.

## Installation

```bash

# from the repo root
uv sync --package flashdreams-wan21

```

## Running the method

To run Wan2.1, launch its v2 T2V application:

This command uses the v2 application.

```bash

uv run --package flashdreams-wan21 \
    flashdreams-run-v2 \
    t2v-wan21-t2v-1.3b-480p \
    --output-path artifacts/t2v-wan21-t2v-1.3b-480p.mp4 -- \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

```

For multi-GPU inference, run the same command under `torchrun` (taking
4 GPUs as an example):

```bash

uv run --package flashdreams-wan21 \
    torchrun --nproc_per_node=4 --no-python flashdreams-run-v2 \
    t2v-wan21-t2v-1.3b-480p \
    --output-path artifacts/t2v-wan21-t2v-1.3b-480p.mp4 -- \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

```

The package also exposes the following pipeline configs for direct use:

| Method | Description |
| --- | --- |
| `wan21-t2v-1.3b-480p` | Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only). |
| `wan21-i2v-14b-480p` | Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame). |

To inspect all supported CLI arguments and their default values, run:

```bash

uv run --package flashdreams-wan21 \
    flashdreams-run-v2 t2v-wan21-t2v-1.3b-480p -- --help

```

Some generated Wan2.1 samples:

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/wan21/wan21-t2v-1.3b-480p.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      prompt: "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    </div>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/wan21/wan21-i2v-14b-480p.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      prompt: "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
      <br/>
      image: https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG
    </div>
  </div>
</div>

## Profiling benchmark

Here is the profiling benchmark on DiT per-step runtime for FlashDreams Wan2.1
compared to the [official Wan2.1 implementation](https://github.com/Wan-Video/Wan2.1)
and the [FastVideo](https://github.com/hao-ai-lab/FastVideo) baseline under
matched settings.

 <figure class="benchmark-figure-wrap">
   <div
     id="wan21-benchmark-chart"
     class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/wan21/perf-0521.json"
     data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="Wan2.1 benchmark chart"
   ></div>
   <figcaption>
    <p class="model-footnote">
       This chart shows per-diffusion-step DiT runtime in milliseconds with CFG at 480p (81 frames) on a single GPU.
       For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings.
       For the official Wan2.1 implementation, see
       <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/wan21/tests/parity_check">this instruction</a>.
       For the FastVideo baseline, see
       <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/wan21/tests/baseline_fastvideo">this instruction</a>.
     </p>
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

## Citation

If you use Wan2.1, please cite the original work:

```bibtex

@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal={arXiv preprint arXiv:2503.20314},
      year={2025}
}
```
