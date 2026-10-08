---
name: flashdreams-integrations
description: Navigate FlashDreams model and application integration boundaries, including the experimental inference API, its higher-level demo API, the separate v2 application protocol, built-in recipes, package layout, registration, and test placement. Use before adding or restructuring code under integrations_v2/, apps/, flashdreams/flashdreams/recipes/, or the runtime/demo adapter boundary.
---

# FlashDreams integration architecture

Use this skill to decide which API family and directory own a change. Read the
referenced implementation before editing it; the contracts are evolving and the
source is authoritative.

For the end-to-end model-porting procedure, also read `integrate-a-model`. For
Python API documentation, use `python-docstring-style`.

## Choose the boundary first

FlashDreams currently has two layers in the experimental runtime stack, plus a
separate v2 application stack:

| Need | Contract | Typical owner |
| --- | --- | --- |
| Reusable, presentation-independent model execution | `flashdreams.runtime` | A model adapter/runtime/session under `integrations_v2/<model>/impl/` |
| Replay, benchmark, local-window, or WebRTC demo orchestration | `flashdreams.runtime.demo` | A demo adapter and model-specific scenario preparation above the inference adapter |
| An application implemented as model/UI loops and run by `flashdreams-run-v2` | `flashdreams.api_v2`, driven by `flashdreams.runtime_v2` | `apps/<name>/` plus a binding under `integrations_v2/<model>/apps/<name>/` |
| Low-level denoising pipeline pieces | `flashdreams.infra` and built-in recipes | `flashdreams/flashdreams/recipes/<name>/` |

Do not mix similarly named sessions, step results, or lifecycle hooks across
these contracts. In particular, `flashdreams.api_v2.ISession` is not
`flashdreams.runtime.InferenceSession`.

Start with the maintained API map in `docs/src/content/docs/documentation/index.md`. Then follow
the narrower source guide for the selected boundary:

- `ARCHITECTURE.md` explains both API families and the v2 threading model.
- `integrations_v2/README.md` describes v2 application packages and registration.
- `flashdreams/flashdreams/api_v2/README.md` and
  `flashdreams/flashdreams/runtime_v2/README.md` define the application protocol
  and its runner.
- `flashdreams/flashdreams/recipes/template/README.md` is the reference only for
  the lower-level `StreamInferencePipeline` contracts. It is not a runtime-API or
  demo-API example.

For experimental inference or demo work, implement the public contracts named
by the API reference and keep model execution separate from scenario,
benchmark, and presentation policy. `integrations_v2/lingbot/impl/runtime.py`
is the current concrete inference adapter example; shared protocol tests live
under `flashdreams/tests/`.

## V2 application protocol

This is a separate API family, retained for applications run by
`flashdreams-run-v2`:

- `IApplication` parses application arguments, owns shared expensive state, and
  creates sessions.
- `ISession` owns one run and registers an `IModelLoop` plus an optional
  `IUILoop`.
- `runtime_v2` owns the window, input/output transport, presentation, and the
  two-thread lifecycle.

Use shared app packages when they already express the interaction:

```text
apps/<name>/                         reusable model-agnostic application
integrations_v2/<model>/            model implementation and config
integrations_v2/<model>/apps/<name>/adapter.py
                                    binding and zero-argument create_app factory
```

An app must depend on `flashdreams`, not on a concrete integration. Its tests
use a stand-in model. The model-specific binding may depend on both the app and
the framework.

Register real application factories through the
`flashdreams.applications_v2` entry-point group in the integration's
`pyproject.toml`. The entry-point slug resolves to a zero-argument `create_app`
or `create_app_<suffix>` factory and appears in `flashdreams-run-v2 --help`.
Arguments before `--` belong to the runtime; arguments after it belong to the
application.

Do not add a legacy `flashdreams.runner_configs` entry point merely to expose a
v2 application.

## Lower-level pipeline recipes

The older pipeline layer is still used by model implementations:

```text
flashdreams.core
  -> flashdreams.infra
       -> flashdreams/flashdreams/recipes/<name>/
       -> integrations_v2/<model>/
```

- `core` owns model-agnostic numerical/distributed primitives and must not
  import `infra`, apps, recipes, or integrations.
- `infra` owns reusable pipeline/config/orchestration contracts and may depend
  on `core`, never on a concrete model.
- `recipes` contain reusable in-tree model components and pipeline configs.
- `integrations_v2/<model>` packages concrete model code and may depend on the
  framework and the app it adapts.

Before changing `StreamInferencePipeline`, `Transformer`, streaming encoder or
decoder, cache, CP, CFG, scheduler, or CUDA-graph behavior, compare the concrete
recipe and its base class. Use
`flashdreams/flashdreams/recipes/template/` for the minimal contract, not as a
template for the newer runtime/demo boundary.

`flashdreams-run` is the legacy recipe-runner CLI. Built-in recipe configs
self-register through `flashdreams.configs.registry`; plugin runners use the
`flashdreams.runner_configs` entry-point group. This mechanism is separate from
both `flashdreams.runtime.demo` and `flashdreams-run-v2`. Extend it only when the
task explicitly targets a recipe runner.

## Placement and tests

Put code next to the contract it validates:

| Change | Tests |
| --- | --- |
| Experimental runtime/demo protocol or shared implementation | `flashdreams/tests/` |
| V2 runtime/protocol behavior | `flashdreams/test_v2/` |
| Model-agnostic app behavior | `apps/<name>/tests/`, against a stand-in model |
| Model implementation or adapter | `integrations_v2/<model>/tests/` |
| Built-in recipe | `flashdreams/tests/` |

Every pytest test needs a `ci_cpu`, `ci_gpu`, or `manual` marker. Keep config,
schema, registration, mapping, and stand-in tests on CPU. Real checkpoints,
CUDA execution, and quality/performance comparisons belong behind `ci_gpu` or
`manual` as appropriate.

For a new integration or binding, verify at minimum:

1. dependency direction and package metadata;
2. schema/config validation without loading a checkpoint;
3. the appropriate registration entry point;
4. session isolation, reset, close, and terminal behavior;
5. a CPU stand-in path, plus an explicitly marked real-model test when needed.

Inspect runner/config resolution without instantiating a GPU model when the
selected stack supports it. Do not run generation, download checkpoints, or
start GPU/WebRTC workflows unless the task explicitly requires them.
