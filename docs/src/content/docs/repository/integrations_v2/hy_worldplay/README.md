---
title: 'HY-WorldPlay'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-hyworldplay-readme--hy-worldplay"></a>

<a id="integrationsv2-hyworldplay-readme--integration-links"></a>

## Integration links

- **Applications:** [Cam2V](apps/cam2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/hy_worldplay/config.py)

HY-WorldPlay is a streaming camera-controlled image-to-video model with action,
camera-trajectory, and reconstituted-context memory conditioning. Its public
model configuration is the `PIPELINE_HY_WORLDPLAY_WAN_I2V_5B` object exported
from `hy_worldplay.config`.

<a id="integrationsv2-hyworldplay-readme--pipeline-configuration"></a>

## Pipeline configuration

- `PIPELINE_HY_WORLDPLAY_WAN_I2V_5B`

The pipeline uses the HY-WorldPlay distilled checkpoint by default and keeps
memory-selection settings on the pipeline config.

<a id="integrationsv2-hyworldplay-readme--install"></a>

## Install

```bash

uv sync --package flashdreams-hy-worldplay --inexact

```

Export `HF_TOKEN` with read access to the gated `tencent/HY-WorldPlay`
repository before the first checkpoint download.

<a id="integrationsv2-hyworldplay-readme--cam2v-application"></a>

## Cam2V application

HY-WorldPlay binds the pipeline to the reusable Cam2V v2 application:

```bash

uv sync --package flashdreams-hy-worldplay --inexact
uv run --no-sync flashdreams-run-v2 cam2v-hy-worldplay \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

Shared [apps/cam2v](../../apps/cam2v/README.md) documentation lists controls,
application arguments, and development commands.

<a id="integrationsv2-hyworldplay-readme--low-level-pipeline-construction"></a>

## Low-level pipeline construction

To construct the model pipeline directly:

```python

from hy_worldplay.config import PIPELINE_HY_WORLDPLAY_WAN_I2V_5B

pipeline = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.setup().to("cuda").eval()

```

<a id="integrationsv2-hyworldplay-readme--tests"></a>

## Tests

```bash

uv run --no-sync pytest integrations_v2/hy_worldplay -m ci_cpu
```
