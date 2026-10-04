---
title: 'FastVideo CausalWan 2.2 T2V application'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-fastvideocausalwan22-apps-t2v-readme--fastvideo-causalwan-2-2-t2v-application"></a>

<a id="integrationsv2-fastvideocausalwan22-apps-t2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared T2V](../../../../apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/fastvideo_causal_wan22/config.py)

```bash

uv run --package flashdreams-fastvideo-causal-wan22 flashdreams-run-v2 \
  t2v-fastvideo-causal-wan2.2-t2v-14b --output-path clip.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7 --no-compile

```

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
