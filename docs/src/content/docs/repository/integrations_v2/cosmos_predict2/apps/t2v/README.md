---
title: 'Cosmos Predict2 T2V application'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-cosmospredict2-apps-t2v-readme--cosmos-predict2-t2v-application"></a>

<a id="integrationsv2-cosmospredict2-apps-t2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared T2V](../../../../apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/cosmos_predict2/config.py)

Cosmos Predict2 generates the complete clip in one bidirectional block, so
`--total-blocks` must remain `1`.

```bash

uv run --package flashdreams-cosmos-predict2 flashdreams-run-v2 \
  t2v-cosmos2-t2v-2b-720p --output-path clip.mp4 -- \
  --prompt "A cat surfing" --no-compile

```

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
