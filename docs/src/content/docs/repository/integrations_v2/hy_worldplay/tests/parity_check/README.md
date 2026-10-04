---
title: 'HY-WorldPlay parity check'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--hy-worldplay-parity-check"></a>

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--integration-links"></a>

## Integration links

- **Applications:** [hy worldplay applications](../../README.md#integrationsv2-hyworldplay-readme--integration-links)
- **Configuration:** [hy worldplay configuration](../../README.md#integrationsv2-hyworldplay-readme--integration-links)

Self-contained benchmark of upstream
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) WAN-5B
I2V model. `run.sh` executes upstream's own `wan/generate.py`, with a
loading-only patch that stages the distilled checkpoint in CPU memory, and
preserves the vendor baseline used while the FlashDreams integration was being
developed.

The current integration is the `cam2v-hy-worldplay` v2 application. It no
longer registers the historical `flashdreams-run hy-worldplay-wan-i2v-5b`
runner, so the native comparison commands and results later on this page are
retained as historical evidence, not as a current runnable workflow. See the
[integration README](../../README.md) for the supported application command.

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--run"></a>

## Run

From `integrations_v2/hy_worldplay/tests/parity_check/`, run:

```bash

bash run.sh

```

Single-GPU defaults are used; override via env vars:

```bash

NUM_GPU=4 NUM_CHUNK=4 POSE='w-16' bash run.sh

```

Other tunables (defaults shown):

| env var | default | meaning |
| --- | --- | --- |
| `NUM_GPU` | `1` | torchrun `--nproc_per_node` |
| `NUM_CHUNK` | `1` | autoregressive chunk count (each chunk = 4 latents) |
| `POSE` | `w-4` | camera trajectory (must total `NUM_CHUNK * 4` latents) |
| `SEED` | `0` | RNG seed |
| `PROMPT` | `&quot;First-person view ... ancient Athens ...&quot;` | text prompt |
| `IMAGE_PATH` | `${REPO_DIR}/assets/img/test.png` | first-frame I2V input |
| `OUTPUT_DIR` | `${REPO_DIR}/outputs/parity` | benchmark output dir |
| `REPO_DIR` | `${SCRIPT_DIR}/HY-WorldPlay` | upstream checkout and checkpoint root |
| `SKIP_HEAVY_DEPS` | `0` | when `1`, skip installing vendor-only dependencies |
| `USE_KV_CACHE_TRUE` | `0` | when `1`, swap vendor&#x27;s `wan/generate.py` for the `use_kv_cache=True` monkey-patch (phase 2b.6 acceptance baseline -- see below) |

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--re-baselining-against-vendor-s-usekvcache-true-code-path"></a>

### Re-baselining against vendor's `use_kv_cache=True` code path

Phase 2b.6 closes the native HY-WorldPlay runner by validating
parity against vendor's *cache-prefill* code path
(`use_kv_cache=True`) rather than the single-forward-pass default.
Set `USE_KV_CACHE_TRUE=1` to swap the default `wan/generate.py`
invocation for `run_vendor_use_kv_cache.py`, which runtime-monkey-
patches `WanPipeline.__setattr__` so `self.use_kv_cache = False`
(vendor's line 707 inside `predict`) is silently coerced to `True`:

```bash

USE_KV_CACHE_TRUE=1 \
    NUM_CHUNK=2 POSE='w-8' SEED=0 \
    OUTPUT_DIR="${PWD}/HY-WorldPlay/outputs/parity_use_kv_cache_true" \
    bash run.sh

```

The output MP4 lands in `${OUTPUT_DIR}`. The historical comparison used the
`imageio` snippet below and accepted `mean |Δ| ≤ 5 / 255`.

This mode is the **2b.6 acceptance baseline**. The default
(no env var) mode keeps producing the phase-1
`use_kv_cache=False` baseline so older parity numbers remain
comparable.

The script is idempotent: on first run it clones upstream, downloads
`tencent/HY-WorldPlay`'s `wan_transformer/` and `wan_distilled_model/`
checkpoints into `HY-WorldPlay/hf_models/`, synchronizes the isolated
environment, installs the vendor-only dependencies, and runs the benchmark.
Subsequent runs skip the checkout and checkpoint downloads but still verify
the environment before re-running the benchmark.

`run.sh` currently sets `PIN_COMMIT=HEAD`: a new checkout uses upstream's
current default-branch head, while an existing checkout is left at its current
commit. Record `git -C HY-WorldPlay rev-parse HEAD` with any new result; the
checked-in phase-1 numbers are historical and are not automatically
reproducible against a newer upstream revision.

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--outputs"></a>

## Outputs

Written under `HY-WorldPlay/outputs/parity/` by default:

- `<pose>_<sanitized_prompt>.mp4` — generated video (16 fps)
- `err.txt` — error log (only created on failures)

`bench.sh`, `bench_batch.sh`, and `bench_pairs.sh` document the historical
matched-input workflow, but still invoke the removed
`flashdreams-run hy-worldplay-wan-i2v-5b` runner and therefore do not run
against the current tree. They remain useful for interpreting the checked-in
benchmark results and would need to be ported to the v2 Cam2V application
before collecting new native-versus-vendor numbers.

The two MP4s should be equivalent (same checkpoint, same pipeline,
same RNG seed). They are **not** bit-for-bit identical because the
plugin and the upstream script run as separate processes against
separate venvs, so they accumulate independent CUDA-stream-ordering
noise, independent autotune-cache state, and independent H.264 encoder
nondeterminism. Compare numerically, not via `cmp`:

```bash

REFERENCE_MP4=/path/to/reference.mp4 uv run python - <<'PY'
import os
import numpy as np, imageio.v3 as iio
from pathlib import Path
a = iio.imread(next(Path("HY-WorldPlay/outputs/parity").glob("*.mp4")))
b = iio.imread(os.environ["REFERENCE_MP4"])
assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
d = np.abs(a.astype(np.int16) - b.astype(np.int16))
print(f"mean |d| : {d.mean():.4f}  (uint8 / 255)")
print(f"max  |d| : {d.max()}        (uint8 / 255)")
print(f"frames with mean |d| > 5: {(d.mean(axis=(1,2,3))>5).sum()}/{a.shape[0]}")
PY

```

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--historical-parity-results"></a>

### Historical parity results

The phase-1 integration measured parity numerically against the upstream
`wan/generate.py` output. On the reference single-GPU `--num-chunk 1
--pose w-4 --seed 0` benchmark, **on the same torch version**, the
observed drift is:

| comparison | mean `|Δ|` (uint8) | max `|Δ|` | frames mean `|Δ| &gt; 5` |
| --- | --- | --- | --- |
| **plugin vs upstream (same torch)** | **3.41** | 130 | **0** |
| upstream vs itself (torch 2.11 vs 2.12) | 3.76 | 138 | 0 |
| plugin vs plugin (two runs, same venv) | 0.00 | 0 | 0 |

So the plugin reproduces upstream more tightly than upstream reproduces
itself across a torch minor bump, and **zero frames cross the
"visually noticeable" mean-delta-of-5 threshold**. The plugin itself
is bit-deterministic across runs in the same venv.

Two prerequisites applied to this historical bar:

1. **`torch` version must match between the parity venv and the
   FlashDreams environment being compared.** The parity `pyproject.toml` pins
   `torch==2.11.*` to mirror the repository-root `uv.lock`. When FlashDreams
   bumps torch, bump this pin in lockstep — otherwise drift jumps from
   ~3.4 to ~5 (we measured a 3.76 contribution from the 2.11 -> 2.12
   minor alone).
2. **The native default prompt had to byte-match upstream's
   `wan/generate.py` `--input` argparse default.** An early
   version had a trailing `.` that shifted the UMT5 tokenisation by
   one token and added ~2 of drift on its own. The current default lives in
   `hy_worldplay/impl/conditioning.py`; the old `runner.py` and its
   `tests/test_smoke.py` guard were removed with the runner.

These results should not be treated as a parity claim for the current Cam2V
application; that application has a different session and presentation path.

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--isolation"></a>

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `HY-WorldPlay/` resolve to *this* venv, not the surrounding
flashdreams one.

The sub-venv includes `flashdreams-hy-worldplay` as a path source so its
Python implementation can be imported alongside the upstream dependencies.
It does not restore the removed phase-1 runner entry point. Outside this
directory, use
`uv run --project integrations_v2/hy_worldplay/tests/parity_check ...`
to target the same environment from elsewhere in the repo.

This isolation keeps HY-WorldPlay's heavy vendor-only dependencies out of the
repository-root `uv.lock`. `run.sh` installs those dependencies into this
environment on demand; set `SKIP_HEAVY_DEPS=1` only when they are already
installed or the vendor leg will not be run.

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--files-tracked-here"></a>

## Files tracked here

- `README.md` — this file
- `run.sh` — supported vendor clone, setup, and benchmark entry point
- `run_vendor_use_kv_cache.py` and patch helpers — vendor parity probes
- `bench*.sh`, `bench*_summary.py`, and checked-in benchmark Markdown —
  historical native-versus-vendor tooling and results
- `pyproject.toml` and `uv.lock` — isolated environment definition and lock
- `.gitignore` — ignores generated checkouts, environments, caches, and output

<a id="integrationsv2-hyworldplay-tests-paritycheck-readme--runtime-requirements"></a>

## Runtime requirements

- Linux or WSL with Bash, `git`, and `uv`.
- NVIDIA GPU with CUDA support. The distilled checkpoint is about 40 GiB and
  `run.sh` loads it through CPU memory to avoid a roughly 48 GiB GPU-loading
  peak; actual inference memory depends on GPU count and chunk count.
- `HF_TOKEN` exported with read access to `tencent/HY-WorldPlay`.
- More than 52 GiB free disk for the downloaded checkpoints, plus space for
  the checkout, environment, outputs, and Hugging Face cache. Budget at least
  ~70 GiB on the target filesystem, and more if the cache duplicates downloads.
- Vendor-only Python dependencies. `run.sh` installs them, including
  `sageattention`; `--use_sageattn false` remains the benchmark default.
