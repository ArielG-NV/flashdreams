---
title: 'CLI and API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Choose an API by what you are building:

| Task | API | Command |
| --- | --- | --- |
| Implement a model adapter with reusable inference sessions | [Experimental inference API](inference_api.md) (`flashdreams.runtime`) | — |
| Add replay, output modes, warmup, or benchmarks to an inference adapter | [Experimental demo API](demo_api.md) (`flashdreams.runtime.demo`) | `flashdreams-run` for registered demos |
| Build a v2 application with model and optional UI loops | [V2 application API](application_api.md) (`flashdreams.api_v2`) | `flashdreams-run-v2` |

The two experimental APIs share a session contract. V2 applications use their
own `ISession` and loop contracts. Legacy runner presets launched with
`flashdreams-run` also expose the lower-level [infra](infra.md) pipeline and
[runner](integrations.md) APIs.

### [CLI](cli.md)

The `flashdreams-run` and `flashdreams-run-v2` commands, slugs,
options, and launch modes.

### [Experimental inference API](inference_api.md)

The presentation-independent `flashdreams.runtime` contracts for
model adapters, reusable runtimes, isolated sessions, inputs, and step
results.

### [Experimental demo API](demo_api.md)

The higher-level `flashdreams.runtime.demo` API for demo scenarios,
input and output modes, drivers, warmup, replay, and benchmarking.

### [V2 application API](application_api.md)

The `flashdreams.api_v2` application, session, model-loop, and optional
UI-loop contracts run by `flashdreams.runtime_v2`.

### [Core](core.md)

The low-level kernels and process-group utilities that
integrations share: attention, the block-structured KV cache, and
distributed helpers.

### [Infra](infra.md)

The swappable building blocks for model pipelines: the
config system, the encoder / diffusion-model / decoder triple, and
the streaming inference pipeline that drives them.

### [Pipelines and runners](integrations.md)

In-tree recipes, legacy runner configs, and the layout of v2 integration
packages.

### [Serving](serving.md)

WebRTC presentation for v2 applications, its transport, and example
launch commands.
