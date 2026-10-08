---
title: 'Architecture'
---

<a id="architecture--architecture"></a>

FlashDreams exposes one public application protocol,
`flashdreams.api_v2`. Model and runtime implementation layers sit behind that
boundary; applications do not depend on their private lifecycle contracts.

## Repository layers

| Layer | Responsibility | Canonical documentation |
| --- | --- | --- |
| `flashdreams.core` | Model-neutral tensor, cache, attention, and distributed primitives. | [Model guide](../documentation/inferencing_api/guides/create_model.md) |
| `flashdreams.infra` and `flashdreams.recipes` | Pipeline configuration and reusable model components. | [Pipeline overview](../documentation/inferencing_api/guides/stream_inference_pipeline.md) and [demo-configuration guide](../documentation/demo_api/guides/configuration.md) |
| `flashdreams.api_v2` | Public application, session, loop, and client-I/O protocols. | [Demo Application API](../documentation/demo_api/api_reference/application.md) |
| `flashdreams.runtime_v2` | Runtime that executes api_v2 applications. | [Runtime internals](flashdreams/flashdreams/runtime_v2/README.md) |
| `apps` | Reusable, model-agnostic api_v2 applications. | [Application packages](apps/README.md) and [demo guide](../documentation/demo_api/guides/create_demo.md) |
| `integrations_v2` | Standalone model packages and adapters that bind models to applications. | [Integration packages](integrations_v2/README.md) and [adapter guide](../documentation/demo_api/guides/integrate_model.md) |

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

The public
[application API](../documentation/demo_api/api_reference/application.md)
defines these
contracts. The repository [runtime v2 notes](flashdreams/flashdreams/runtime_v2/README.md) document buffering, threading,
resets, distributed execution, and shutdown behavior for maintainers.

For the model-side lifecycle inside a generation step, see the
[inference pipeline overview](../documentation/inferencing_api/guides/stream_inference_pipeline.md).

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
