---
title: 'Causal Wan2.2'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

[Model weights](https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers)
[Official code](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_t2v.py)

CausalWan2.2 is a [FastVideo](https://github.com/hao-ai-lab/FastVideo)-released
14B MoE causal-diffusion variant of Wan 2.2 with 8-step inference.

This integration uses `flashdreams-run-v2`.

See the [Causal Wan 2.2 integration reference](../repository/integrations_v2/fastvideo_causal_wan22/README.md) for its registered
configuration and application wiring.

## Requirements

- **Minimum VRAM**: ~112 GB.
- **PyTorch**: >= 2.9.

## Installation

```bash

# from the repo root
uv sync --package flashdreams-fastvideo-causal-wan22 --inexact

```

## Running the method

To run Causal Wan2.2, launch its v2 T2V application. For example:

```bash

uv run --no-sync \
    flashdreams-run-v2 \
    t2v-fastvideo-causal-wan2.2-t2v-14b \
    --output-path artifacts/t2v-fastvideo-causal-wan2.2-t2v-14b.mp4 -- \
    --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
    --total-blocks 7

uv run --no-sync \
    flashdreams-run-v2 \
    t2v-fastvideo-causal-wan2.2-t2v-14b \
    --output-path artifacts/t2v-fastvideo-causal-wan2.2-t2v-14b-raccoon.mp4 -- \
    --prompt "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar." \
    --total-blocks 7

```

For multi-GPU inference, run the same command under `torchrun` (taking
4 GPUs as an example):

```bash

uv run --no-sync \
    torchrun --nproc_per_node=4 --no-python flashdreams-run-v2 \
    t2v-fastvideo-causal-wan2.2-t2v-14b \
    --output-path artifacts/t2v-fastvideo-causal-wan2.2-t2v-14b.mp4 -- \
    --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
    --total-blocks 21

```

The package exposes the following pipeline config for direct use:

| Method | Description |
| --- | --- |
| `fastvideo-causal-wan2.2-t2v-14b` | FastVideo CausalWan 2.2 14B MoE T2V (Wan VAE decoder, 8-step). |

To inspect all supported CLI arguments and their default values, run:

```bash

uv run --no-sync \
    flashdreams-run-v2 t2v-fastvideo-causal-wan2.2-t2v-14b -- --help

```

Some generated samples from the above commands:

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      prompt: "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
    </div>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      prompt: "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
    </div>
  </div>
</div>

## Citation

FastVideo lists the following research citations:

```bibtex

@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}

@article{zhang2025vsa,
  title={Vsa: Faster video diffusion with trainable sparse attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Huang, Haofeng and Lin, Will and Liu, Zhengzhong and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.13389},
  year={2025}
}
```
