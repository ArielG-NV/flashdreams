---
title: 'Self-Forcing parity check'
---

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--self-forcing-parity-check"></a>

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--integration-links"></a>

## Integration links

- **Applications:** [self forcing applications](../../README.md#integrationsv2-selfforcing-readme--integration-links)
- **Configuration:** [self forcing configuration](../../README.md#integrationsv2-selfforcing-readme--integration-links)

Self-contained benchmark of upstream
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) with a small local
patch (`changes.patch`) that adds `EventProfiler`-based per-block timing
and JSON stats output (mirroring `flashdreams`'s pipeline profiling). The
benchmark forces PyTorch's cuDNN attention backend.

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--requirements"></a>

## Requirements

- Bash, Git, and `uv`
- Network access to clone Self-Forcing and download the model checkpoints
- A CUDA-capable NVIDIA GPU environment with cuDNN attention support

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--run"></a>

## Run

From this directory, run:

```bash

bash run.sh

```

That's it. The script is idempotent: on first run it clones upstream at a
pinned commit, downloads `Wan-AI/Wan2.1-T2V-1.3B` and the
`self_forcing_dmd.pt` checkpoint, applies `changes.patch`, and runs the
benchmark. Subsequent runs skip whatever's already in place and just
re-run the benchmark.

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--outputs"></a>

## Outputs

Written under `Self-Forcing/`:

- `videos/offline.mp4` — generated video
- `videos/stats_offline.json` — per-block timings (`denoise_ms`,
  `kv_update_ms`, `decode_ms`, `total_ms`, `total_ms_wo_finalize`) plus
  GPU memory stats, one entry per autoregressive block

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--isolation"></a>

## Isolation

Dependencies are declared in this directory's `pyproject.toml`, locked in
`uv.lock`, and installed into `./.venv/`. Because `uv run` walks upward
looking for a project, calls from inside `Self-Forcing/` resolve to *this*
environment, not the surrounding flashdreams one.

<a id="integrationsv2-selfforcing-tests-paritycheck-readme--files-tracked-here"></a>

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + patch + benchmark, idempotent
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `uv.lock` — resolved dependency lock file
- `changes.patch` — local edits on top of the pinned upstream commit
  (`EventProfiler` timing, JSON stats dump, route attention through the
  `attention()` dispatcher so `FORCE_CUDNN_ATTN=1` works end-to-end)
- `.gitignore` — ignores the cloned `Self-Forcing/` tree and `./.venv/`
