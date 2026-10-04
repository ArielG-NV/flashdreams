---
title: 'Lingbot World'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-lingbot-readme--lingbot-world"></a>

<a id="integrationsv2-lingbot-readme--integration-links"></a>

## Integration links

- **Applications:** [Cam2V](apps/cam2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/lingbot/config.py)

Lingbot World is a streaming camera-controlled image-to-video model integration.
Its public model variants are `StreamInferencePipelineConfig` literals in
`lingbot.config`; the reusable Cam2V application owns interactive I/O.

<a id="integrationsv2-lingbot-readme--pipeline-configurations"></a>

## Pipeline configurations

- `PIPELINE_LINGBOT_WORLD_FAST`
- `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3`
- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST`
- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3`

The installed Cam2V application slugs select these configurations as follows:

| Application slug | Pipeline config |
| --- | --- |
| `cam2v-lingbot` | `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3` |
| `cam2v-lingbot-world-fast` | `PIPELINE_LINGBOT_WORLD_FAST` |
| `cam2v-lingbot-world-fast-taehv-window15-sink3` | `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3` |
| `cam2v-lingbot-world-v2-14b-causal-fast` | `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST` |
| `cam2v-lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` | `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3` |

`cam2v-lingbot` is the short compatibility alias for the bounded-window TAEHV
default.

<a id="integrationsv2-lingbot-readme--install"></a>

## Install

```bash

uv sync --package flashdreams-lingbot --inexact

```

Checkpoints download from Hugging Face on first use. Export `HF_TOKEN` when the
selected repository requires authentication.

<a id="integrationsv2-lingbot-readme--cam2v-application"></a>

## Cam2V application

```bash

uv sync --package flashdreams-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

Shared [apps/cam2v](../../apps/cam2v/README.md) documentation lists controls,
application arguments, and development commands.

<a id="integrationsv2-lingbot-readme--low-level-pipeline-construction"></a>

## Low-level pipeline construction

```python

from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST

pipeline = PIPELINE_LINGBOT_WORLD_FAST.setup().to("cuda").eval()

```

<a id="integrationsv2-lingbot-readme--tests"></a>

## Tests

```bash

uv run --no-sync pytest integrations_v2/lingbot -m ci_cpu
```
