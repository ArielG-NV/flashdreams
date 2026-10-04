---
title: 'Wan 2.2 TI2V application'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-wan22-apps-t2v-readme--wan-2-2-ti2v-application"></a>

<a id="integrationsv2-wan22-apps-t2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared T2V](../../../../apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/wan22/config.py)

Launch the single-block Wan 2.2 application with a prompt and first frame.
`--image-path` is required and must point to an existing image:

The example below uses a local input image. Replace the placeholder path with
the first frame that should establish the generated clip.

```bash

uv run --package flashdreams-wan22 flashdreams-run-v2 \
  t2v-wan22-ti2v-5b --output-path clip.mp4 -- \
  --prompt "A cat surfing" \
  --image-path /path/to/first-frame.png \
  --no-compile

```

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
