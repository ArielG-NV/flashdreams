---
title: 'Serving'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

This page covers WebRTC presentation for [v2 applications](application_api.md). See the [API overview](index.md) for the separate runtime
families.

## Serving building blocks

- **Application** implements the `flashdreams.api_v2` application and session
  protocols and declares its application-specific arguments.
- **Model loop** manages model lifecycle and cached state across steps.
- **Runtime** owns the session threads, presentation pacing, and selected client
  window. `--mode webrtc` selects the browser client window; `--host` and
  `--port` configure its server.
- **V2 WebRTC transport** under
  `flashdreams/flashdreams/runtime_v2/serving/` handles HTTP, signaling,
  browser input, and media transport.
- **Integration adapter** owns model-specific checkpoint setup, conditioning,
  prompt/scene semantics, and chunk generation, then registers an application
  slug with the v2 runtime.

## Reference integration

The reusable applications provide concrete examples of the v2 serving stack:

- shared v2 transport code under
  `flashdreams/flashdreams/runtime_v2/serving/`,
- camera-to-video application behavior under `apps/cam2v/`,
- thin model bindings under `integrations_v2/<model>/apps/cam2v/adapter.py`,
- OmniDreams model wiring under `integrations_v2/omnidreams/` and serving
  under `apps/interactive_drive/`.

The separate `flashdreams/flashdreams/serving/webrtc/` package supports the
experimental demo API.

## Launch patterns

Single GPU:

```bash

uv run flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc -- --example-data --total-blocks 21

```

Multi GPU:

```bash

uv run torchrun --nproc_per_node=2 --no-python flashdreams-run-v2 \
    cam2v-lingbot --mode webrtc -- --example-data --total-blocks 21

```

## See also

- [demo_api](demo_api.md)
- [inference_api](inference_api.md)
- [/developer_guides/interactive_serving](../developer_guides/interactive_serving.md)
- [/models/lingbot_world](../models/lingbot_world.md)
