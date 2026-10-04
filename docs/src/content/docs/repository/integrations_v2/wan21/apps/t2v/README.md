---
title: 'Wan 2.1 T2V application'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-wan21-apps-t2v-readme--wan-2-1-t2v-application"></a>

<a id="integrationsv2-wan21-apps-t2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared T2V](../../../../apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/wan21/config.py)

```bash

uv run --package flashdreams-wan21 flashdreams-run-v2 \
  t2v-wan21-t2v-1.3b-480p --output-path clip.mp4 -- \
  --prompt "A cat surfing" --no-compile

```

Wan 2.1 generates the complete clip in one bidirectional rollout, so
`--total-blocks` must remain `1` (the default).

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
