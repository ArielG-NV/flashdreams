---
title: 'Causal-Forcing T2V application'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-causalforcing-apps-t2v-readme--causal-forcing-t2v-application"></a>

<a id="integrationsv2-causalforcing-apps-t2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared T2V](../../../../apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/causal_forcing/config.py)

```bash

uv run --package flashdreams-causal-forcing flashdreams-run-v2 \
  t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise --output-path clip.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7 --no-compile

```

For the framewise config, replace the application name with
`t2v-causal-forcing-wan2.1-t2v-1.3b-framewise`.

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
