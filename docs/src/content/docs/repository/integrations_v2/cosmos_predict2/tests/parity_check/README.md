---
title: 'Cosmos-Predict2.5 parity check'
---

<a id="integrationsv2-cosmospredict2-tests-paritycheck-readme--cosmos-predict2-5-parity-check"></a>

<a id="integrationsv2-cosmospredict2-tests-paritycheck-readme--integration-links"></a>

## Integration links

- **Applications:** [cosmos predict2 applications](../../../../../models/cosmos_predict2.md#developer-details)
- **Configuration:** [cosmos predict2 configuration](../../../../../models/cosmos_predict2.md#developer-details)

GPU-only reproducer of upstream
[`nvidia-cosmos/cosmos-predict2.5`](https://github.com/nvidia-cosmos/cosmos-predict2.5)
T2V base-model inference, used as a numerical and qualitative reference
for the in-tree `cosmos_predict2` FlashDreams integration.

It requires Bash, Git, curl, uv, an NVIDIA GPU, and enough disk space for
the upstream environment and model weights. The first inference run also
downloads the model weights from Hugging Face Hub.

<a id="integrationsv2-cosmospredict2-tests-paritycheck-readme--run"></a>

## Run

From `integrations_v2/cosmos_predict2/tests/parity_check/`, run:

```bash

bash run.sh

```

That's it. Idempotent: on first run it

1. clones `cosmos-predict2.5` at the pinned commit,
2. checks out that commit,
3. applies `changes.patch` on top (skipped if already applied),
4. re-fetches the three LFS-tracked input assets the command needs,
5. materializes `cosmos-predict2.5/.venv/` via `uv sync --extra=cu130`,
6. runs the T2V inference command.

Subsequent runs reuse the clone, patch, assets, and environment, then
re-run T2V inference. An I2V command remains commented out in `run.sh`;
it is intentionally disabled for an apples-to-apples T2V performance
comparison.

<a id="integrationsv2-cosmospredict2-tests-paritycheck-readme--outputs"></a>

## Outputs

Written under `cosmos-predict2.5/`:

- `outputs/base_text2world/` — T2V result for `assets/base/robot_welding.json`

<a id="integrationsv2-cosmospredict2-tests-paritycheck-readme--isolation"></a>

## Isolation

We use cosmos-predict2.5's own `pyproject.toml` verbatim — `uv sync
--extra=cu130` materializes the venv at `cosmos-predict2.5/.venv` and
pulls torch 2.9.1 + cu130 plus the matching prebuilt
`flash-attn` / `decord` / `transformer-engine` / `natten` wheels from
NVIDIA's custom index. None of those wheels are ABI-compatible with
flashdreams' `torch>=2.11` pin, so we deliberately keep this venv
independent of flashdreams' rather than stacking on top of it. The
in-tree `cosmos_predict2` flashdreams integration is exercised separately
via the main `flashdreams/.venv`.

<a id="integrationsv2-cosmospredict2-tests-paritycheck-readme--files-tracked-here"></a>

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + checkout + patch + LFS fetch + `uv sync` + T2V inference, idempotent
- `changes.patch` — local edits layered on top of the pinned upstream commit:

  - defer dataset imports and skip automatic experiment-module loading so
    unrelated config registration does not load unavailable backends,
  - add opt-in torch SDPA, cuDNN attention, and `torch.compile` controls used
    by `run.sh` for the parity measurement
- `.gitignore` — ignores the cloned `cosmos-predict2.5/` tree (which
  carries its own `.venv/`, lockfile, outputs, etc.)
