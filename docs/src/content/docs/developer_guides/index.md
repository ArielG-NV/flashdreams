---
title: 'Guides'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

### [Inference pipeline overview](inference_pipeline_overview.md)

The end-to-end computation flow: cache initialization, autoregressive
generation steps, and finalization.

### [Config system](config_system.md)

How typed dataclass configs compose, how inference and demo configs
differ, and how to apply nested CLI overrides.

### [Application slugs and model adapter dispatch](runner_slugs.md)

How public v2 application slugs are registered, how reusable apps and
model-owned adapters divide responsibilities, and how to add a binding.

### [Add a new method](new_integration.md)

How to structure a standalone integration, implement its pipeline and
runner, register its entry point, and add its documentation.

### [Local demo benchmarks](local_benchmarks.md)

How to run command-backed local benchmarks that capture logs, MP4s,
metrics, environment metadata, and an HTML report.

### [Interactive serving](interactive_serving.md)

Choose the demo or v2 serving stack and launch WebRTC presentation.

### [OmniDreams latency tuning](latency_tuning.md)

Select a supported OmniDreams preset, resolution, transport, and
measurement path.

### [flashdreams.accelerated](flashdreams_accelerated.md)

The low-level quantization and optimized multi-head attention building
blocks used to accelerate streaming video models.

## Where these guides fit

These guides are conceptual. For a specific method, see its per-model
page under [/models/index](../models/index.md). For Python interfaces, see the [API overview](../api/index.md). For the shortest path from installation to a generated
clip, see [/quickstart/index](../quickstart/index.md).
