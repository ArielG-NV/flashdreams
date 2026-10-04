---
title: 'Interactive serving'
---

:orphan:

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams keeps a world-model session alive while inputs and outputs stream
through the application loop. See the [API overview](../api/index.md) for the
two serving stacks.

## Serving models

<div class="fd-highlight-grid">
  <div class="fd-highlight-card">
    <div class="fd-highlight-title">Live input</div>
    <div class="fd-highlight-body">Application controls or sensor updates arrive continuously.</div>
  </div>
  <div class="fd-highlight-card">
    <div class="fd-highlight-title">Warm session</div>
    <div class="fd-highlight-body">Pipeline and cache state persist across updates.</div>
  </div>
  <div class="fd-highlight-card">
    <div class="fd-highlight-title">Model step</div>
    <div class="fd-highlight-body">Encoder, transformer, scheduler, and decoder advance the world.</div>
  </div>
  <div class="fd-highlight-card">
    <div class="fd-highlight-title">Streamed output</div>
    <div class="fd-highlight-body">Frames or latent output return without closing the session.</div>
  </div>
</div>

## Reference integrations

- [/models/omnidreams](../models/omnidreams.md) shows closed-loop autonomous-vehicle simulation
  through the v2 `interactive-drive-omnidreams` application.
- [/models/lingbot_world](../models/lingbot_world.md) shows camera control through the v2
  `cam2v-lingbot` application.
- [/quickstart/index](../quickstart/index.md) provides the shortest command-level path for
  trying inference and serving side by side.

Launch WebRTC presentation for a demo API application with:

```bash

uv run flashdreams-run DEMO_SLUG \
    --output webrtc --host 0.0.0.0 --port 8089

```

Launch a v2 application with:

```bash

uv run flashdreams-run-v2 APPLICATION_SLUG \
    --mode webrtc --host 0.0.0.0 --port 8089

```

For v2, runtime options precede `--` and application-specific options follow
it. Use `flashdreams-run --help` or `flashdreams-run-v2 --help` to list the
installed slugs for the corresponding API family.

## Serving implementation references

- [/api/serving](../api/serving.md) for serving API concepts and component mapping.
- [/api/demo_api](../api/demo_api.md) for the experimental demo API and its WebRTC output
  specification.
- [/developer_guides/inference_pipeline_overview](inference_pipeline_overview.md) for runner/pipeline
  execution flow.
- `flashdreams.serving.webrtc` for the demo API family's shared WebRTC
  server, session manager, runtime protocol, data-channel messages, packaged
  UI app factory, and distributed serve-loop helpers.
- `flashdreams.runtime_v2.serving` for the separate v2 HTTP, signaling,
  browser-input, and media transport.
- `apps/interactive_drive` for the reusable OmniDreams application loops;
  model-specific bindings live under
  `integrations_v2/omnidreams/apps/`.

## Demo API WebRTC shape

For the experimental demo API family, the shared package owns the reusable
serving shell:

- `runtime.WebRTCSessionRuntime` describes the lifecycle a model runtime
  must provide: initialize, reset a session, declare the next step, execute it,
  and close.
- `manager.BaseWebRTCSessionManager` owns one active peer connection, control
  data-channel parsing, liveness, keyboard resampling, loopback warmup, and
  chunk scheduling.
- `demo.serve_webrtc_demo` joins a `~flashdreams.runtime.demo.WebRTCOutputSpec`,
  session manager, and browser resources into the shared server.
- `server.create_packaged_webrtc_app` builds the aiohttp app from packaged
  browser assets and keeps those resources alive until cleanup.
- `bootstrap.initialize_cuda_distributed` and `bootstrap.run_webrtc_server`
  provide the common CUDA/distributed launch and teardown path.
- `messages` defines the common action, event, heartbeat, disconnect,
  `chunk_done`, `event_ack`, and error payload contracts.

Model integrations should keep model-specific runtime code local: checkpoint
setup, scene or prompt semantics, conditioning/rendering, cache math, and any
custom HTTP routes that configure a session. The shared WebRTC layer should be
enough for a new demo to provide a runtime, package its UI assets, add optional
routes, and start serving without copying another demo's bootstrap code.

V2 applications instead implement `IApplication`, `ISession`,
`IModelLoop`, and optionally `IUILoop`. Their WebRTC presentation is
selected by `flashdreams-run-v2 --mode webrtc`; they should not implement the
demo API's `WebRTCSessionRuntime` contract.
