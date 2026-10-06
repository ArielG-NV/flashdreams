---
title: 'Principle Agent Guide'
---

<a id="agents--principle-agent-guide"></a>

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

Bug fix = root cause, not symptom: a report names a symptom. Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins, but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size, lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a known ceiling (global lock, O(n²) scan, naive heuristic) with a `ponytail:` comment naming the ceiling and upgrade path.

Not lazy about: understanding the problem (read it fully and trace the real flow before picking a rung, a small diff you don't understand is just laziness dressed up as efficiency), input validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs (the platform is never the spec ideal, a clock drifts, a sensor reads off), anything explicitly requested. Lazy code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (an assert-based demo/self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.

<a id="agents--flashdreams-agent-guide"></a>

## FlashDreams Agent Guide

FlashDreams is a GPU-heavy inference and serving library for autoregressive video and world models. Default to inspection, docs, config checks, and CPU tests unless the user explicitly asks to run generation or GPU workflows.

Start here, then use the narrower docs for the task in front of you:

- `skills/` contains repo-authored Agent Skills. Use
  [Adding an Agent Skill](CONTRIBUTING.md#contributing--adding-an-agent-skill) for local
  setup and authoring rules.
- [Get Started](../quickstart/index.md) covers setup and a first run;
  [Models](../models/index.md) lists supported models.
- `CONTRIBUTING.md` covers PR process, DCO sign-off, coding conventions, test markers, and dependency rules.
- [Adding documentation](CONTRIBUTING.md#contributing--adding-documentation) covers the
  Zensical site. [Testing](CONTRIBUTING.md#contributing--testing) covers local, Docker, and
  CI test entry points.

<a id="agents--repo-map"></a>

## Repo Map

File structure is in [CONTRIBUTING.md's File Tree Of FlashDreams](CONTRIBUTING.md#contributing--file-tree-of-flashdreams). Ignore gitignored AI-tool directories (see `.gitignore`) when scanning the source tree; they hold tool state, not the repo's current source.

<a id="agents--skill-map"></a>

## Skill Map

- `skills/adding-documentation`: read before adding, moving, or substantially editing documentation or a repository Agent Skill.
- `skills/flashdreams-integrations`: read before changing recipe/integration architecture, runner registration, pipeline wiring, or model boundaries.
- `skills/integrate-a-model`: read before porting a new external model or reproducing an integration workflow.
- `skills/profile-model-performance`: read before starting performance work on an existing model integration, demo, runner, or serving path; use it to map execution, add stage timings, and identify decode/model/cache/transfer/presentation bottlenecks.
- `skills/apply-inference-optimizations`: read before porting runtime speedups such as bounded K/V caches, overlap, compile, CUDA graphs, decoder layout changes, or presentation queue tuning into an integration.
- `skills/validate-performance-quality`: read before adding benchmark sweeps, quality comparisons, profiler probes, performance summaries, or docs for a performance change.
- `skills/flashdreams-postprocessing`: read before adding or modifying video post-processors, postprocess presets, `VideoPostprocessStream`, buffering/layout behavior, or runner postprocess wiring.
- `skills/python-docstring-style`: read before adding or polishing Python docstrings, field docstrings, module comments, or SPDX headers.
- `skills/maintaining-oss-state`: read before dependency, license, NOTICE, REUSE, or OSS-release collateral changes.
- When adding a new `skills/<skill-name>/SKILL.md`, update this section so agents can discover when to use it.

<a id="agents--common-commands"></a>

## Common Commands

```bash

uv sync --extra dev --extra runners
uv sync --package <integration> --extra dev
uv run flashdreams-run --help
uv run flashdreams-run <runner-name> --help
uv run flashdreams-run --no-instantiate <runner-name>
uv run --group lint pre-commit run -a
uv run --group test pytest -m ci_cpu
uv run --group test pytest -m "not manual"
./tests/run_tests_local.sh [target]
./tests/run_tests_docker.sh [target]
uv run --only-group docs zensical build -f docs/zensical.toml
uv run --only-group docs zensical serve -f docs/zensical.toml

```

Use `--no-instantiate` before GPU work to inspect the resolved runner config without constructing models or loading checkpoints.

<a id="agents--no-gpu-workflow"></a>

## No-GPU Workflow

- Inspect available runners with `uv run flashdreams-run --help`, then inspect a specific runner with `uv run flashdreams-run --no-instantiate <runner-name>`.
- Prefer CPU checks first: config imports, checkpoint key-remap shape/bijection tests on CPU or meta tensors, docs builds, `pytest -m ci_cpu`, and static assertions about runner names and pipeline wiring.
- Avoid `ci_gpu`, generation, `torchrun`, Docker GPU tests, large Hugging Face downloads, rollout parity, CUDA graph, WebRTC runtime, and quality-regression tests on CPU-only hosts unless the user requests them or the test explicitly skips cleanly.

<a id="agents--testing-guidance"></a>

## Testing Guidance

Every pytest test must carry a `ci_cpu`, `ci_gpu`, or `manual` marker; `manual` may override a module-level CI marker, but `ci_cpu` and `ci_gpu` must not be combined. `CONTRIBUTING.md` has the exact rules, including how pytest discovers files and functions (`test_*.py`, `test_*`). Use module-level `pytestmark = pytest.mark.ci_cpu` for pure Python/metadata tests. Keep GPU, `libGL`/`cv2`, large-checkpoint, credential, and download-heavy checks out of `ci_cpu`.

**v2 test ownership** — put a new test next to the thing it validates:

| You are testing… | Test lives in… |
| --- | --- |
| FlashDreams Runtime/Protocol (window, threads, presentation) | `flashdreams/test_v2/` |
| An app (flags, WASD, physics) | `apps/&lt;name&gt;/tests/` |
| A model or its adapter | `integrations_v2/&lt;model&gt;/tests/` |

<a id="agents--dependencies"></a>

## Dependencies

- `infra` depends on `core` — never the other way around. `core` stays model-agnostic.
- `recipes`/`integrations_v2` depend on `infra` and `core` — never the other way around. Expose a generic config slot or override hook in `core`/`infra` instead of adding model-specific branches. Built-in reusable model pieces belong in `flashdreams/flashdreams/recipes/`; standalone plugin packages belong in `integrations_v2/<name>/`.
- `apps/<name>/` depends on `flashdreams` — never the other way around. An app is written against the framework, not against any one model: it must run against a stub network, and binding a real model is the adapter's job.
- `integrations_v2/<model>/` depends on `flashdreams` and on the app it adapts for (via its own `integrations_v2/<model>/apps/<demo>/adapter.py`) — never the other way around.

Because of this direction, tests in `apps/<name>/tests/` must not import from `integrations_v2/` — an app's tests run against a stub, and model-specific checks belong in `integrations_v2/<model>/tests/`. CI enforcement of this is a separate follow-up, not yet built.

<a id="agents--known-pitfalls"></a>

## Known Pitfalls

- FlashDreams targets large NVIDIA GPUs; many real rollouts need 80 GB VRAM and CUDA-capable dependencies.
- Full `uv sync`/`uv run` can build CUDA packages such as Transformer Engine. For narrow work, use `uv sync --package <integration> --extra dev` when possible.
- `flashdreams-run --no-instantiate` resolves config only; it does not prove weights, CUDA execution, or parity.
- First GPU runs include compile/autotune warmup, so benchmark harnesses should discard warmup chunks.
- Docs CI mocks heavy GPU packages; use the commands in
  [Adding documentation](CONTRIBUTING.md#contributing--adding-documentation).
- Plain `pytest` includes manual tests. Use `pytest -m ci_cpu` or `pytest -m "not manual"` for normal local validation.
- New or moved skills must pass the Agent Skills frontmatter checks in `tests/test_agent_skills.py`.

<a id="agents--troubleshooting-links"></a>

## Troubleshooting Links

- Setup and requirements: `README.md`, `docs/src/content/docs/quickstart/index.md`
- CLI details: `docs/src/content/docs/documentation/cli.md`
- Integration/plugin layout: `docs/src/content/docs/api/integrations.md`
- New integrations: `docs/src/content/docs/developer_guides/new_integration.md`
- Docs and CPU autodoc: [Adding documentation](CONTRIBUTING.md#contributing--adding-documentation)
- Tests and quality regressions: [Testing](CONTRIBUTING.md#contributing--testing)
- Security reports: `SECURITY.md`

<a id="agents--api-surfaces"></a>

## API Surfaces

- `flashdreams.runtime` is the experimental inference API: model adapters create
  reusable `InferenceRuntime` objects, which create isolated `InferenceSession`
  objects. `flashdreams.runtime.demo` is the demo API above that boundary; it
  owns input/output modes, drivers, warmup, replay, benchmarking, and demo
  validation without moving those concerns into model inference.
- `flashdreams.api_v2` is the public application protocol API implemented by
  apps and integrations (`IApplication`, `ISession`, `IModelLoop`, and optional
  `IUILoop`). `flashdreams.runtime_v2` runs those protocols and provides the
  `flashdreams-run-v2` command. Do not mix its session/loop contracts with the
  experimental `flashdreams.runtime` session contract.

<a id="agents--v2-threading"></a>

### v2 threading

Each `run_session` process uses two session threads:

- The calling main/UI thread collects user events, manages the client window,
  advances/composites presentation, and drives the UI loop.
- The model thread drives the model loop and publishes generated output.
