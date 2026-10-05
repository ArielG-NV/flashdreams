---
title: 'HY-WorldPlay Cam2V'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-hyworldplay-apps-cam2v-readme--hy-worldplay-cam2v"></a>

<a id="integrationsv2-hyworldplay-apps-cam2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared Cam2V](../../../../apps/cam2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/hy_worldplay/config.py)

Install the HY-WorldPlay integration and launch its `cam2v-hy-worldplay`
application:

```bash

uv sync --package flashdreams-hy-worldplay --inexact
uv run --no-sync flashdreams-run-v2 cam2v-hy-worldplay \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

See the shared [Cam2V README](../../../../apps/cam2v/README.md) for controls,
application arguments, defaults, and tests. The application uses
`PIPELINE_HY_WORLDPLAY_WAN_I2V_5B`; see the
[HY-WorldPlay integration README](../../../../../models/hy_worldplay.md#developer-details) for model details.
