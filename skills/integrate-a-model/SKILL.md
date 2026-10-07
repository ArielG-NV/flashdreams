---
name: integrate-a-model
description: End-to-end workflow for porting an external video or world model into FlashDreams: choose the experimental inference API, its higher-level demo API, or the separate v2 application protocol; reuse a pipeline recipe; map checkpoints; add model deltas; package the integration; and validate CPU structure, GPU execution, parity, and performance. Use when integrating a new model, porting upstream weights, or reproducing an existing integration.
---

# Integrate a model into FlashDreams

Use this procedure to port an external model. Read
`skills/flashdreams-integrations/SKILL.md` first for API ownership, package
layout, registration, and test placement. Use `docs/src/content/docs/api/index.md` for
the canonical API map, and read `python-docstring-style` before writing public
Python documentation.

Current references serve different purposes:

- `integrations_v2/hy_worldplay/` is a complete pipeline port with
  model-specific conditioners and a v2 Cam2V application binding.
- `integrations_v2/lingbot/impl/runtime.py` is the current inference-adapter
  example.
- `flashdreams/flashdreams/recipes/template/README.md` documents only the
  lower-level streaming-pipeline contract.

## Choose the public boundary first

Choose the boundary with `skills/flashdreams-integrations/SKILL.md`; do not
rederive it here. Record the selected public contract and execution surface in
the integration plan before scaffolding files.

## Phase 0 — Scope before estimating

Record these facts from the upstream repository and model overview:

1. **Backbone family.** Find the closest existing recipe or integration. A
   Wan/DiT derivative should usually reuse Wan components rather than port a
   second network.
2. **Checkpoint format.** Record repository IDs, gated-access requirements,
   file format, shard index, envelope keys, dtype, and whether the published
   weights are native or converted.
3. **Inference behavior.** Record resolution, temporal shape, scheduler,
   number of steps, guidance, one-shot versus autoregressive execution, and
   cache policy.
4. **Model deltas.** Identify extra conditioning, attention, memory, control,
   or decoder behavior relative to the closest base.
5. **Consumer boundary.** Decide whether the deliverable needs inference only,
   a demo over that inference API, a v2 application, or more than one explicit
   adapter.
6. **Parity source.** Pin the upstream commit, environment, inputs, seed, and
   expected comparison metric before changing implementation details.

If the architecture is novel, say so: checkpoint plumbing may still be small,
but the network and inference-loop phases will not be.

## Phase 1 — Scaffold the integration package

An in-tree integration is a workspace member under
`integrations_v2/<model>/`; the root `pyproject.toml` includes
`integrations_v2/*`. Follow the current layout:

```text
integrations_v2/<model>/
├── __init__.py
├── config.py                 # public pipeline-config literal(s)
├── impl/                     # all model-specific implementation
├── tests/                    # model and adapter tests
├── apps/                     # only for v2 application bindings
│   └── <app>/
│       ├── adapter.py
│       └── README.md
├── README.md
└── pyproject.toml
```

Keep implementation out of the integration root except for `config.py`. An
out-of-tree integration may use the same package shape and depend on a released
`flashdreams` instead of the workspace source.

The version in an in-tree package is synchronized to
`flashdreams/flashdreams/_version.py` by the `sync-version` pre-commit hook.
Run the hook rather than hand-maintaining a divergent version.

Registration depends on the selected boundary:

- Experimental inference and demo adapters currently have no package
  entry-point group in this repository. Export and compose the adapter where
  its Python consumer needs it.
- A v2 application registers a zero-argument factory under
  `flashdreams.applications_v2`:

  ```toml
  [project.entry-points."flashdreams.applications_v2"]
  "cam2v-my-model" = "my_model.apps.cam2v.adapter:create_app"
  ```

  The factory returns an uninitialized `IApplication`; heavyweight setup
  belongs in the application lifecycle, not import time.
- Add `flashdreams.runner_configs` only when the requested deliverable is the
  legacy `flashdreams-run` recipe-runner surface.

## Phase 2 — Reuse the closest pipeline recipe

In `config.py`, copy or derive the closest base pipeline and replace only the
pieces that differ: encoder, transformer network, scheduler, decoder, or
checkpoint transform. Export named module-level config literals. HY-WorldPlay,
for example, derives `PIPELINE_HY_WORLDPLAY_WAN_I2V_5B` from
`PIPELINE_WAN22_TI2V_5B` and keeps its implementation under `impl/`.

When constructing a specialized dataclass config, copy base fields explicitly
when omission should fail loudly. Preserve and verify model-defining fields
such as temporal length, cache window, guidance, image-latent stamping,
precision, compilation, and CUDA-graph settings. A distilled checkpoint may
also require a different scheduler and exact published timesteps.

Do not create a builder abstraction just to avoid one config literal. Use a
builder only when the model genuinely has dynamic variants; inspect
`integrations_v2/flashvsr/` for that larger case.

## Phase 3 — Load and remap the checkpoint

Prefer the checkpoint whose parameter structure is closest to the code being
ported. Native, Diffusers, FSDP, and training envelopes often name identical
tensors differently, so verify instead of assuming one format is better.

Use the existing checkpoint loader and remap helpers under
`flashdreams.core.checkpoint`; do not write a second downloader or shard
loader. `load_checkpoint` supports ordinary PyTorch files and sharded
safetensors indexes. A state-dict transform should unwrap known envelopes,
remove known training prefixes, apply ordered renames, and leave tensors
unchanged.

Before loading a large checkpoint on a GPU, prove that transformed checkpoint
keys and shapes match the model:

```python
model = {name: tuple(value.shape) for name, value in network.state_dict().items()}
checkpoint = {
    name: tuple(value.shape)
    for name, value in state_dict_transform(raw_meta_state_dict).items()
}

missing = set(model) - set(checkpoint)
extra = set(checkpoint) - set(model)
shape_mismatches = {
    name: (model[name], checkpoint[name])
    for name in model.keys() & checkpoint.keys()
    if model[name] != checkpoint[name]
}
assert not missing and not extra and not shape_mismatches
```

Construct the network and placeholder tensors on the `meta` device when the
model supports it. For safetensors, read shapes from shard headers with
`safe_open(...).get_slice(name).get_shape()`; for a PyTorch file, use
`torch.load(..., map_location="meta", weights_only=True)` when the format
supports meta loading. Run the integration's actual transform, not a duplicate
test-only mapping.

Add a `ci_cpu` test for deterministic transforms and representative real key
strings. If checking every header requires a gated download, keep that check
`manual` rather than silently downloading tens of gigabytes in CPU CI.

When changing the default checkpoint source, compare the two fully transformed
state dicts tensor by tensor. Key/shape equality proves structural
compatibility; tensor equality proves the source conversion itself. A missing
parameter is usually a naming or envelope mismatch, so diff names before
assuming the publisher omitted weights.

## Phase 4 — Add only model-specific deltas

Each genuine delta should remain in `integrations_v2/<model>/impl/`. Reuse
`flashdreams.core`, `flashdreams.infra`, and recipe hooks; never add
model-specific branches to those shared layers.

For residual conditioners, zero-initialize new residual heads when that is part
of the upstream design so the unloaded delta is an identity. If a base
checkpoint is intentionally supported, tolerate only the precisely enumerated
new keys when loading it; keep all unrelated missing and unexpected keys
strict. Verify cache, reset, and history ownership per session rather than on a
global model object.

## Phases 5–7 — Implement the selected public boundary

Follow `skills/flashdreams-integrations/SKILL.md` and the corresponding page
under `docs/src/content/docs/api/`. Keep the binding thin:

- inference adapters own reusable model execution and isolated rollout state;
- demo adapters add scenario and presentation policy above inference;
- v2 bindings reuse a model-agnostic app under `apps/` and register a
  zero-argument factory through `flashdreams.applications_v2`.

Use `integrations_v2/lingbot/impl/runtime.py` as the current inference example
and HY-WorldPlay's Cam2V adapter as the current v2 example. For v2, verify:

```bash
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 <app-slug> -- --help
```

Arguments before `--` belong to the runtime; arguments after it belong to the
application.

## Phase 8 — Verify from cheapest to most expensive

1. **CPU structure and unit tests.** Test package metadata, public config
   literals, schema validation, checkpoint transforms, factory registration,
   reset/close behavior, and a stand-in pipeline. Mark every pytest test
   `ci_cpu`, `ci_gpu`, or `manual`.
2. **No-instantiation inspection.** For a legacy recipe runner only, use
   `uv run flashdreams-run --no-instantiate <slug>`. For v2, inspect help and
   test the application with a stand-in model; the v2 command has no equivalent
   model-config-only guarantee.
3. **Checkpoint load.** On a suitable GPU, load the real weights and ensure no
   parameters remain on `meta` and no unapproved keys are missing or extra.
4. **Short rollout.** Run the selected inference, demo, or v2 surface with the
   smallest valid temporal extent and verify shapes, lifecycle, and output.
5. **Upstream parity.** Pin both environments and compare the same checkpoint,
   input, seed, dtype, scheduler, attention backend, frame count, and decoder.
   Report the metric and acceptance threshold; do not inherit a historical
   threshold from another integration.
6. **Performance.** Discard compile/autotune warmup, report per-stage and
   end-to-end medians, and match software stacks before claiming a speedup.
   Read `profile-model-performance` and `validate-performance-quality` before
   adding a benchmark or publishing results.

For an in-tree FlashDreams change, finish with the repository-required checks:

```bash
uv run --group lint pre-commit run -a
uv run --group test pytest -m ci_cpu
```

These are CPU checks. Do not start checkpoint downloads, GPU generation,
WebRTC, or vendor parity runs on a CPU-only host.

## Documentation deliverables

Update the integration `README.md` with installation, credentials, hardware
requirements, and the supported execution surface. If there is a v2 binding,
keep launch-only details in `apps/<app>/README.md`. If there is an inference or
demo adapter, name it explicitly and link the corresponding API guide; never
present a low-level `pipeline.setup()` example as an inference session.

For a user-visible model page, mirror the current pages under
`docs/src/content/docs/models/` and register it in `docs/src/content/docs/models/index.md`. State
which API family each command uses. Keep historical benchmark scripts and
removed runner commands out of current quickstarts.

## Common pitfalls

- The root workspace glob is `integrations_v2/*`, not `integrations/*`.
- Current v2 model bindings use `flashdreams.applications_v2` and
  `flashdreams-run-v2`; HY-WorldPlay no longer registers its historical
  `flashdreams-run hy-worldplay-wan-i2v-5b` runner.
- Recheck `docs/src/content/docs/api/index.md` rather than inferring ownership from
  similarly named runtime modules.
- Diffusers repositories may be sharded. Point the loader at the index when
  appropriate rather than inventing a single-file URL.
- First-run timings include compile and autotune work. Never report them as
  steady state.
- Full `uv sync` or `uv run` may build CUDA extensions. Prefer the narrow
  package command documented by the integration and CPU-safe tests first.
- Keep checkpoints, upstream source trees, generated videos, and benchmark
  outputs out of the package and Git history.

## Done criteria

- [ ] The chosen inference, demo, v2, or legacy-runner boundary is explicit.
- [ ] The integration reuses the closest recipe and keeps model code under
      `integrations_v2/<model>/impl/`.
- [ ] Checkpoint keys and shapes are a full, explained match.
- [ ] CPU config/schema/registration/lifecycle tests pass.
- [ ] A real-checkpoint load and short GPU rollout pass on suitable hardware.
- [ ] Upstream parity is measured against a pinned, documented baseline.
- [ ] Demo concerns live in `flashdreams.runtime.demo`, not in the inference
      session.
- [ ] A v2 application, if present, uses `flashdreams.applications_v2` and does
      not masquerade as an inference/demo adapter.
- [ ] User docs contain only commands supported by the current tree.
- [ ] Repository lint and `ci_cpu` checks pass.

## Evaluating this skill

Test the procedure in an isolated branch or worktree that contains this skill
and the intended base recipe but not the target integration. Confirm those
preconditions before launching the evaluation. Start with the highest-signal
GPU-free slice: select the base recipe, implement the checkpoint transform,
and prove the key/shape match. Then score API-boundary choice, package layout,
session lifecycle, parity, and test coverage before attempting GPU performance
work. Do not let the evaluator read a removed reference integration or Git
history containing the answer.
