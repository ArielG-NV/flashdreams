---
title: 'FastVideo causal Wan2.2 T2V parity check'
---

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--fastvideo-causal-wan2-2-t2v-parity-check"></a>

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--integration-links"></a>

## Integration links

- **Applications:** [fastvideo causal wan22 applications](../../README.md#integrationsv2-fastvideocausalwan22-readme--integration-links)
- **Configuration:** [fastvideo causal wan22 configuration](../../README.md#integrationsv2-fastvideocausalwan22-readme--integration-links)

Self-contained benchmark of upstream [FastVideo](https://github.com/hao-ai-lab/FastVideo)
for the self-forcing causal Wan2.2 text-to-video (T2V) path, aligned with
flashdreams parity conventions.

This harness:

- clones upstream FastVideo at a pinned commit,
- applies `changes.patch`,
- uses an isolated `uv` env with FastVideo-compatible deps, then installs local
  `flashdreams` (no-deps) so the benchmark can import `flashdreams.infra.profiler`,
- runs a benchmark script that emits parity-style per-block timing JSON.

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--requirements"></a>

## Requirements

- Bash, Git, and [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA CUDA GPU with enough memory for the 14B model
- Network access for the initial Git clone, Python dependencies, and model download

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--run"></a>

## Run

From this directory:

```bash

bash run.sh

```

The script reuses an existing clone at the pinned commit and does not reapply an
already-applied patch. It runs `uv sync` and reinstalls the local editable
`flashdreams` package on every invocation; `uv` reuses already-satisfied packages.

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--outputs"></a>

## Outputs

Written under `FastVideo/`:

- `videos/offline.mp4` - generated video
- `videos/stats_offline.json` - timing JSON per autoregressive block, including
  `autoregressive_index`, `denoise_ms`, `kv_update_ms`, `total_ms`,
  `total_ms_wo_finalize`, and CUDA memory fields

The patched denoising stage does not time video decoding. It writes
`decode_ms: 0.0` in each block only for parity-schema compatibility.

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--backend-and-speed-settings"></a>

## Backend and speed settings

The benchmark runs with:

- CPU offload disabled (`dit/text_encoder/vae` offload all `False`)
- `--enable_torch_compile` enabled
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- `FASTVIDEO_FORCE_CUDNN_SDPA=1`

This enforces a fast-path configuration and applies strict cuDNN SDPA forcing
inside FastVideo's SDPA backend when the env flag is set.

`flashdreams` is installed with `uv pip install --no-deps -e ...` to expose the
profiler module without forcing flashdreams' full dependency set into this
FastVideo parity environment.

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--notes"></a>

## Notes

The benchmark uses the default raccoon-and-sunflowers prompt embedded in the
patched benchmark script. Override it by running that script directly with
`--prompt`.

<a id="integrationsv2-fastvideocausalwan22-tests-baselinefastvideo-readme--files-tracked-here"></a>

## Files tracked here

- `run.sh` - clone + setup + patch + benchmark runner
- `pyproject.toml` - isolated venv definition
- `uv.lock` - locked dependency versions
- `changes.patch` - local edits on top of pinned FastVideo commit
- `.gitignore` - ignores local clone and venv
