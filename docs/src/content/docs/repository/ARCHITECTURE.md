---
title: 'Architecture'
---

<a id="architecture--architecture"></a>

FlashDreams separates model inference, demo orchestration, and v2 application
execution. The [API overview](../api/index.md) is the canonical map of those
three families; do not mix their session or lifecycle contracts.

## Repository layers

| Layer | Responsibility | Canonical documentation |
| --- | --- | --- |
| `flashdreams.core` | Model-neutral tensor, cache, attention, and distributed primitives. | [Core API](../api/core.md) |
| `flashdreams.infra` | Pipeline, runner, configuration, checkpoint, and component wiring. | [Infrastructure API](../api/infra.md) |
| `flashdreams.runtime` and `flashdreams.runtime.demo` | Reusable inference sessions and higher-level demo orchestration. | [Inference API](../api/inference_api.md) and [demo API](../api/demo_api.md) |
| `flashdreams.api_v2` and `flashdreams.runtime_v2` | Application/session/loop protocols and the runtime that executes them. | [Application API](../api/application_api.md) and [runtime internals](flashdreams/flashdreams/runtime_v2/README.md) |
| `apps` | Reusable, model-agnostic v2 applications. | [Application packages](apps/README.md) |
| `integrations_v2` | Standalone model packages and adapters that bind models to applications. | [Integration packages](integrations_v2/README.md) |

Dependencies point downward: `core` remains model-agnostic; `infra` may
depend on `core`; applications depend on the framework; integrations depend
on the framework and applications. Framework packages must not import a model
integration, and application tests must use stubs rather than import an
integration.

## Execution model

The v2 CLI resolves a registered application, initializes it, creates a
session, and lets the runtime own the client window and lifecycle. Each session
registers one model loop and optionally a UI loop. The main/UI thread handles
input and presentation; the model thread owns model state and generation.
`StepResult` objects carry generated output from the model side to the
presentation side.

The public [application API](../api/application_api.md) defines these
contracts. The repository [runtime v2 notes](flashdreams/flashdreams/runtime_v2/README.md) document buffering, threading,
resets, distributed execution, and shutdown behavior for maintainers.

For the model-side lifecycle inside a generation step, see the
[inference pipeline overview](../developer_guides/inference_pipeline_overview.md).

<a id="architecture--many-gpu-sessions"></a>

## Distributed sessions

In distributed v2 runs, one process is launched per GPU. Rank zero owns the
client window and presentation; every rank owns a model thread and participates
in the integration's tensor/context-parallel collectives. Runtime-level step
agreement keeps admission, input, reset, replacement, and shutdown decisions
consistent across ranks. Integrations remain responsible for their inference
collectives and for gathering complete output before rank zero publishes it.

See the [runtime v2 maintainer notes](flashdreams/flashdreams/runtime_v2/README.md) for the detailed ordering and
failure contracts.
