---
title: 'FlashVSR parity check'
---

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--flashvsr-parity-check"></a>

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--integration-links"></a>

## Integration links

- **Applications:** [flashvsr applications](../../../../../models/flashvsr.md#developer-details)
- **Configuration:** [flashvsr configuration](../../../../../models/flashvsr.md#developer-details)

Self-contained benchmark of upstream
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) with a small local
patch (`changes.patch`) that adds `EventProfiler`-based per-chunk timing
and JSON stats output (mirroring `flashdreams`'s pipeline profiling).

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--run"></a>

## Run

Requirements: Bash, Git, and `uv`; network access for the pinned GitHub
checkout and Hugging Face weights; and an NVIDIA CUDA GPU for the benchmark
and numerical parity checks. The state-dict shape checks themselves are
CPU-only.

From this directory — i.e.

```text

cd integrations_v2/flashvsr/tests/parity_check/

```

run:

```bash

bash run.sh

```

The script manages its own Python environment: on first run
it clones upstream at a pinned commit, downloads
`JunhaoZhuang/FlashVSR-v1.1`, materializes the parity-check venv
(`./.venv/`) — including `pytest` and the workspace's `flashvsr`
package layered on as an editable install so the candidate side is
importable from here — applies `changes.patch`, runs the benchmark, and
then runs both parity tests (`test_tcdecoder_parity.py` and
`test_dit_parity.py`, see below). Subsequent runs reuse the checkout,
downloaded weights, and venv, reset the checkout to the pinned commit,
reapply the patch, and rerun the benchmark and tests.

Override the input video and upscale factor with environment variables:

```bash

INPUT_PATH=/abs/path/to/clip.mp4 SCALE=4.0 bash run.sh

```

`INPUT_PATH` defaults to upstream's `examples/WanVSR/inputs/example4.mp4`.
`SCALE` defaults to `4.0`, matching the patched `benchmark.py` default; set
it to the same value on the FlashDreams side when comparing outputs or stats.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--outputs"></a>

## Outputs

Written under `FlashVSR/examples/WanVSR/`:

- `videos/offline.mp4` — generated upsampled video
- `videos/stats_offline.json` — per-chunk timings (`projector_ms`,
  `dit_ms`, `decoder_ms`, `color_ms`, `total_ms`, `total_ms_wo_finalize`)
  plus GPU memory stats, one entry per autoregressive chunk

The upstream profiler covers its four observable stages: `projector`, `dit`,
`decoder`, and `color`. The FlashDreams pipeline records the more granular
`pad`, `bicubic`, `projector`, `dit_concat`, `denoise`, `decoder`, and `color`
events in `flashvsr.impl.pipeline.FlashVSRPipeline.generate`, plus `finalize`
in the inherited finalization path. Compare the shared stage names directly;
the upstream `dit_ms` is an aggregate rather than a one-to-one match for the
FlashDreams DiT-related events.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--parity-tests"></a>

## Parity tests

`run.sh` invokes both parity tests in this directory after the
benchmark; all three run from the same parity-check venv where `flashvsr`
is layered on as an editable install (`flashdreams-flashvsr =
{ path = "../.." }` in this directory's `pyproject.toml`). To run
the tests manually after `run.sh` has completed setup:

```bash

cd integrations_v2/flashvsr/tests/parity_check/
uv run pytest test_tcdecoder_parity.py test_dit_parity.py -v

```

Individual tests auto-skip when `FlashVSR/` isn't cloned or the relevant
checkpoint isn't staged under `$FLASHVSR_WEIGHTS_ROOT/FlashVSR-v1.1/`.
The numerical chunk-parity and CUDA-graph cases also skip when no GPU is
available; the state-dict shape checks do not require a GPU.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--tc-decoder-parity-testtcdecoderparity-py"></a>

### TC decoder parity (`test_tcdecoder_parity.py`)

Loads upstream's `examples/WanVSR/utils/TCDecoder.py` (out of the
cloned `FlashVSR/` sibling) and the live
`flashvsr.impl.decoder.network.FlashVSR_TAEHV` candidate side-by-side, then
asserts:

- state-dict shapes match `TCDecoder.ckpt` for both reference and
  candidate (after the candidate's `decoder.<i>` → `decoder.blocks.<i>`
  remap),
- chunk-by-chunk numerical parity at fp32 cross-algorithm conv
  tolerance (`atol=2.5e-3 / rtol=1e-3`; see the inline comment in
  `test_tcdecoder_chunk_parity` for why bit-for-bit isn't reachable --
  legacy runs convs at `batch=1` while the candidate runs them at
  `batch=b*t*stride`, so cuDNN picks different kernels); the legacy
  side runs `decode_video(parallel=False)` because upstream's
  `parallel=True` path doesn't carry mem across calls,
- the candidate's CUDA-graph wrapper captures by chunk 4 and matches
  the eager path bit-for-bit (`atol=rtol=1e-5`; both sides share the
  same impl, only the launch path differs).

The upstream file is self-contained (only `torch` / `einops` / `tqdm`
/ stdlib), so it's loaded via `importlib.util.spec_from_file_location`
rather than through `diffsynth.*` -- no package import plumbing
required.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--dit-parity-testditparity-py"></a>

### DiT parity (`test_dit_parity.py`)

Loads upstream's `diffsynth.models.wan_video_dit.WanModel` plus the
streaming-forward wrapper
`diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video` and the
live `flashvsr.impl.transformer.FlashVSRTransformer` candidate side-by-side,
then asserts:

- state-dict shapes match the downloaded
  `FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors`
  checkpoint for both upstream and candidate; the upstream model config
  is derived with `WanModelStateDictConverter().from_civitai(...)`,
- steady-state chunk-by-chunk numerical parity under the streaming
  KV-cache protocol with a calibrated bf16 envelope (`max_abs <= 2.5e-1`
  and `mean_abs <= 3.5e-2`); the upstream cold-start chunk seeds the
  candidate's self-attention KV cache directly because upstream executes
  cold start as one 6-latent-frame call while FlashDreams' production path
  splits work into 2-latent-frame internal steps. The wider envelope
  accounts for upstream's fp64 RoPE + public sparse-attention wrapper
  versus FlashDreams' fused RoPE + direct sparse-attention path.

The upstream `WanModel.forward` is training-only; the streaming
inference path the upsampler actually drives is
`model_fn_wan_video(dit, ...)`, which consumes `dit.patchify` /
`dit.freqs` / `dit.blocks` / `dit.head` / `dit.unpatchify` directly
and is what `FlashVSRTinyLongPipeline` calls per chunk. The test
mirrors that call site verbatim so a future refactor that lands
streaming into `WanModel.forward` will surface as a parity break here.

Unlike the TC decoder file, the DiT references reach back into
`diffsynth.models` / `diffsynth.pipelines`. It is therefore loaded as a
plain package import from the editable `diffsynth` install configured in
`run.sh`.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--sparse-attention-baseline-dependency"></a>

## Sparse attention baseline dependency

Upstream FlashVSR imports `block_sparse_attn` for Locality-Constrained
Sparse Attention. The parity-check venv installs the upstream external
package so the baseline keeps the implementation it originally has. The
FlashDreams candidate uses the in-tree Triton sparse-attention backend
through `flashvsr.impl.transformer.network`.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--isolation"></a>

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `FlashVSR/` resolve to *this* venv, not the surrounding
flashdreams one. The cloned upstream tree itself is registered with
`uv pip install -e ./FlashVSR` so the upstream `diffsynth` / `utils`
packages are importable.

`run.sh` exports `UV_PROJECT_ENVIRONMENT=${SCRIPT_DIR}/.venv` for the
parity-check uv calls. This overrides any inherited shared-workspace setting,
so `uv sync` from here cannot manage that shared venv and uninstall workspace
integrations that this directory's `pyproject.toml` doesn't declare.

Both parity tests and the benchmark run from this same parity-check
venv. The candidate side (`flashvsr.impl.transformer.FlashVSRTransformer`,
`flashvsr.impl.decoder.network.FlashVSR_TAEHV`) is made importable by the
`flashdreams-flashvsr = { path = "../.." }` editable source declared
in `pyproject.toml`; the legacy upstream side is registered via
`uv pip install --no-deps -e ./FlashVSR`. `pytest` is a direct dependency of
this venv, so no workspace-venv flip is needed.

<a id="integrationsv2-flashvsr-tests-paritycheck-readme--files-tracked-here"></a>

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + patch + benchmark + parity tests, reusing setup
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `changes.patch` — local edits on top of the pinned upstream commit
  (`EventProfiler` timing, JSON stats dump)
- `test_tcdecoder_parity.py` — TC decoder parity test against
  upstream's `examples/WanVSR/utils/TCDecoder.py`; run via this
  directory's parity-check venv (see above)
- `test_dit_parity.py` — DiT parity test against upstream's
  `diffsynth.models.wan_video_dit.WanModel` +
  `diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video`; same
  venv as above
- `.gitignore` — ignores the cloned `FlashVSR/` tree and `./.venv/`
