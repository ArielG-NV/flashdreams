---
title: 'FlashVSR V2V'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-flashvsr-apps-v2v-readme--flashvsr-v2v"></a>

<a id="integrationsv2-flashvsr-apps-v2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared V2V](../../../../apps/v2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/flashvsr/config.py)

This adapter binds the shared
[V2V](../../../../apps/v2v/README.md) application to three
FlashVSR configurations.

| Entry-point slug | Configuration |
| --- | --- |
| &lt;code&gt;v2v-flashvsr-v1.1-sparse-ratio-2.0&lt;/code&gt; | Stable sparse attention. |
| &lt;code&gt;v2v-flashvsr-v1.1-sparse-ratio-1.5&lt;/code&gt; | Faster sparse attention. |
| &lt;code&gt;v2v-flashvsr-v1.1-full-attn&lt;/code&gt; | Dense full attention. |

Launch any slug with the v2 application runner:

```bash

uv sync --package flashdreams-flashvsr --inexact
uv run --no-sync flashdreams-run-v2 v2v-flashvsr-v1.1-sparse-ratio-2.0 --output-path upscaled.mp4 -- --video-path input.mp4

```

Omit <code>--video-path</code> to use the bounded Big Buck Bunny default. See
the shared app README for controls, application arguments, presentation modes,
and its CPU test command.
