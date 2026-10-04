---
title: 'OmniDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-omnidreams-readme--omnidreams"></a>

<a id="integrationsv2-omnidreams-readme--integration-links"></a>

## Integration links

- **Applications:** [Interactive Drive](apps/interactive_drive/README.md) and [Crazy Robotaxi](apps/crazy_robotaxi/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/omnidreams/config.py)

OmniDreams is an HD-map-conditioned streaming driving world-model integration.
Its public model variants are `OmnidreamsPipelineConfig` literals in
`omnidreams.config`; reusable applications own interactive I/O.

<a id="integrationsv2-omnidreams-readme--requirements"></a>

## Requirements

- Python 3.10-3.12
- PyTorch 2.11 or newer
- An NVIDIA CUDA GPU; the default Interactive Drive configuration needs about
  48 GB of VRAM

<a id="integrationsv2-omnidreams-readme--pipeline-configurations"></a>

## Pipeline configurations

- `OMNIDREAMS_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_RTX_5090_PIPELINE_CONFIG`
- `OMNIDREAMS_RTX_5090_FAST_PIPELINE_CONFIG`
- `OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG`

<a id="integrationsv2-omnidreams-readme--install"></a>

## Install

```bash

uv sync --package flashdreams-omnidreams --inexact

```

Checkpoints and example scenes download from Hugging Face on first use. Export
`HF_TOKEN` when the selected repository requires authentication.

Configurations that require native DiT or VAE acceleration, including the
`perf` and RTX 5090 variants, download pinned third-party sources on first use into
`artifacts/omnidreams/thirdparty`. Set
`FLASHDREAMS_OMNIDREAMS_TRY_THIRDPARTY_RESYNC=1` to attempt a clean redownload
on each load. Do not use resync while another OmniDreams process is using the
sources. The default value is `0` (only downloads missing sources).

<a id="integrationsv2-omnidreams-readme--applications"></a>

## Applications

- [Interactive Drive](apps/interactive_drive/README.md)
- [Crazy Robotaxi](apps/crazy_robotaxi/README.md)

Launch Interactive Drive in a browser with:

```bash

uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089

```

<a id="integrationsv2-omnidreams-readme--programmatic-pipeline-access"></a>

## Programmatic pipeline access

```python

from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG

pipeline = OMNIDREAMS_PIPELINE_CONFIG.setup().to("cuda").eval()

```

<a id="integrationsv2-omnidreams-readme--tests"></a>

## Tests

```bash

uv sync --package flashdreams-omnidreams --extra dev --group test --inexact
uv run --no-sync pytest integrations_v2/omnidreams/tests -m ci_cpu
```
