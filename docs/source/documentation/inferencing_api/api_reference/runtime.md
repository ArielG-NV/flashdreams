---
title: 'Experimental Inference Runtime API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams.runtime` is the experimental inference API. Its lifecycle is
`ModelAdapter` → reusable `InferenceRuntime` → isolated
`InferenceSession`. See the [Demo API overview](../../demo_api/index.md) for
the application layer above it, or the
[Inferencing API overview](../index.md) for the surrounding model-side guides
and references.

## Minimal direct use

Integrations provide the adapter and its model-specific input schema.

```python
from flashdreams.runtime import InferenceConfig, InferenceInput, ModelAdapter


def run_one_step(adapter: ModelAdapter, global_conditioning, step):
    config = InferenceConfig(model_id=adapter.model_id)
    adapter.validate_config(config)
    runtime = adapter.create_runtime(config)
    try:
        initial_input = InferenceInput(global_conditioning=global_conditioning)
        session = runtime.start_session(initial_input)
        try:
            request = session.next_step_request()
            return None if request is None else session.step(InferenceInput(step=step))
        finally:
            session.close()
    finally:
        runtime.close()
```

Close sessions and runtimes even after errors. Session calls are sequential;
the caller owns concurrency unless an integration says otherwise.

## Runtime lifecycle

::: flashdreams.runtime.InferenceConfig

::: flashdreams.runtime.ModelAdapter

::: flashdreams.runtime.InferenceRuntime

::: flashdreams.runtime.InferenceSession

## Inputs and results

::: flashdreams.runtime.InferenceInput

::: flashdreams.runtime.InferenceInputSchema

::: flashdreams.runtime.StepRequest

::: flashdreams.runtime.StepResult

## Input mapping

::: flashdreams.runtime.CanonicalInputs

::: flashdreams.runtime.InputMapping

::: flashdreams.runtime.InputMappingSchema
