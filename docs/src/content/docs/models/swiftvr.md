---
title: 'SwiftVR'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

[Project page](https://h-oliday.github.io/SwiftVR/)
[arXiv paper](https://arxiv.org/abs/2606.09516)
[Checkpoint](https://huggingface.co/H-oliday/SwiftVR)
[Official code](https://github.com/H-oliday/SwiftVR)

SwiftVR is a real-time, one-step streaming video-restoration model. It combines
mask-free shifted-window attention with a restoration-aware autoencoder for
causal chunk-wise inference. FlashDreams provides 2x and 4x post-processing
presets and the standalone `v2v-swiftvr` application.

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="../_static/model_clips/swiftvr/swiftvr-2x.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<p class="model-footnote">
  FlashDreams SwiftVR 2x output at 2560x1280. The input was the complete
  81-frame, 1280x640 Wan 2.2 sample on this site.
</p>

## Run with FlashDreams

From the repository root:

```bash
uv sync --package flashdreams-swiftvr --inexact
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode mp4 --output-path artifacts/swiftvr-2x.mp4 -- \
  --video-path docs/src/content/docs/_static/model_clips/wan22/wan22-ti2v-5b.mp4
```

## Developer details

[Integration source](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/swiftvr) · [Pipeline configuration](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/swiftvr/config.py) · [Application guide](../repository/integrations_v2/swiftvr/apps/v2v/README.md) · [Tests](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/swiftvr/tests)

### Configurations and behavior

The registered post-processing presets are:

| Preset | Description |
| --- | --- |
| `swiftvr-2x` | Startup-friendly eager 2x restoration. |
| `swiftvr-2x-compiled` | Compiled 2x restoration for long-running streams. |
| `swiftvr-4x` | Eager 4x restoration. |

### Requirements

- **GPU**: One CUDA-capable NVIDIA GPU.
- **Python**: >= 3.10.
- **PyTorch**: >= 2.9.
- **FFmpeg**: Required for MP4 input and output.

### Citation

If you use SwiftVR, please cite the original work:

```bibtex

@article{yan2026swiftvr,
  title={SwiftVR: Real-Time One-Step Generative Video Restoration},
  author={Yan, Jiaqi and Chen, Xiangyu and Zhong, Xinlin and Huang, Haibin and Zhang, Chi and Liu, Jie and Zhou, Jiantao and Li, Xuelong},
  journal={arXiv preprint arXiv:2606.09516},
  year={2026}
}
```
