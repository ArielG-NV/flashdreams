---
title: 'SANA-WM_bidirectional'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

[Project page](https://nvlabs.github.io/Sana/)
[arXiv paper](https://arxiv.org/abs/2410.10629)
[Checkpoint](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional)
[Official code](https://github.com/NVlabs/Sana)

`SANA-WM_bidirectional` is the full-sequence, camera-controlled
[NVlabs/Sana](https://github.com/NVlabs/Sana) world model release. Given a
first frame, a text prompt, and a camera trajectory, it renders a video clip in
a single bidirectional pass. FlashDreams exposes it through the
`PIPELINE_SANA_WM_BIDIRECTIONAL` pipeline configuration, with a native Stage-1 DiT and an LTX-2 refiner.

See the [SANA-WM integration reference](../repository/integrations_v2/sana_wm/README.md) for the programmatic pipeline and
the separately registered streaming application.

The sibling streaming release has a separate model card:
[sana_wm_streaming](sana_wm_streaming.md).

<img alt="SANA-WM bidirectional FlashDreams sample clip." src="../_static/model_clips/sana_wm/sana-wm-bidirectional.avif" />

## Requirements

- **PyTorch**: >= 2.9.
- **Precision**: BF16 by default. The pipeline configuration also exposes opt-in
  FP8 and FP4 execution paths, but the upstream-vs-FlashDreams benchmark for
  `SANA-WM_bidirectional` is BF16-only because upstream
  `SANA-WM_bidirectional` does not support those precision flags.

## Installation

```bash

# from the repo root
uv sync --package flashdreams-sana-wm --inexact

```

## Programmatic pipeline access

The bidirectional model is available as a pipeline configuration:

```python

from sana_wm.config import PIPELINE_SANA_WM_BIDIRECTIONAL

pipeline = PIPELINE_SANA_WM_BIDIRECTIONAL.setup().to("cuda").eval()

```

This is the integration's lower-level pipeline configuration.

## Profiling benchmark

The BF16 chart below compares steady-state in-process generation latency per
generated clip for FlashDreams `SANA-WM_bidirectional` and the official
`SANA-WM_bidirectional` implementation under matched settings on one NVIDIA
GB300 GPU. FlashDreams measured 34,182.39 ms per clip versus 56,932.83 ms for
the official implementation.

In this chart, `Official Impl` means the pinned NVlabs/Sana upstream
implementation measured by the FlashDreams benchmark harness under matched
settings. It is not the SANA-WM 80-scene benchmark result published by the
model authors.

 <figure class="benchmark-figure-wrap">
   <div
     id="sana-wm-bidirectional-bf16-benchmark-chart"
     class="benchmark-figure"
     data-benchmark-json-url="../../_static/performance/sana_wm_bidirectional/perf-0801-bf16.json"
     data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="SANA-WM bidirectional BF16 benchmark chart"
   ></div>
   <figcaption>
     <p class="model-footnote">
       This chart shows steady-state in-process generation latency per generated clip in milliseconds for a
       121-frame full-pipeline BF16 run (Stage-1 DiT + LTX-2 refiner + SANA VAE decode).
       The measured row used one NVIDIA GB300 GPU, one live warmup generation,
       and three measured generations.
       Model construction, checkpoint loading, video writing, and frame dumps are outside the timing boundary.
       The benchmark runs recorded FlashDreams commit bd0816e and upstream commit 6298508.
     </p>
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

## Citation

If you use SANA-WM, please cite the original SANA work:

```bibtex

@misc{xie2024sana,
      title={SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers},
      author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2410.10629},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
