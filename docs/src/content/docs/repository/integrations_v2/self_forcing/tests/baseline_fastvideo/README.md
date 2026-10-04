---
title: 'FastVideo Self-Forcing parity check'
---

<a id="integrationsv2-selfforcing-tests-baselinefastvideo-readme--fastvideo-self-forcing-parity-check"></a>

<a id="integrationsv2-selfforcing-tests-baselinefastvideo-readme--integration-links"></a>

## Integration links

- **Applications:** [self forcing applications](../../README.md#integrationsv2-selfforcing-readme--integration-links)
- **Configuration:** [self forcing configuration](../../README.md#integrationsv2-selfforcing-readme--integration-links)

Self-contained benchmark of upstream [FastVideo](https://github.com/hao-ai-lab/FastVideo) for
the Self-Forcing causal model path, aligned with flashdreams parity-check conventions.

This harness:
- clones upstream FastVideo at a pinned commit,
- applies `changes.patch`,
- uses an isolated `uv` environment with the local `flashdreams` package and the dependencies locked here,
- runs a benchmark script that emits parity-style per-block timing JSON.

It requires Bash, Git, `uv`, network access for the FastVideo clone and model download, and a
CUDA-capable NVIDIA GPU environment that supports the forced cuDNN SDPA backend. The default model
is `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`.

<a id="integrationsv2-selfforcing-tests-baselinefastvideo-readme--run"></a>

## Run

From this directory:

```bash

bash run.sh

```

The script reuses an existing clone, pinned checkout, applied patch, and `uv` environment when they
are already satisfied. It still runs `uv sync` on every invocation so the environment matches the
lock file.

<a id="integrationsv2-selfforcing-tests-baselinefastvideo-readme--outputs"></a>

## Outputs

Written under `FastVideo/`:
- `videos/offline.mp4` - generated video
- `videos/stats_offline.json` - per-block timings with parity-style fields (`denoise_ms`, `kv_update_ms`, `decode_ms`, `total_ms`, `total_ms_wo_finalize`)

<a id="integrationsv2-selfforcing-tests-baselinefastvideo-readme--backend-and-speed-settings"></a>

## Backend and speed settings

The benchmark runs with:
- CPU offload disabled (`dit/text_encoder/vae` offload all `False`)
- `--enable_torch_compile` enabled
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- `FASTVIDEO_FORCE_CUDNN_SDPA=1`

so DiT timing is measured on a fast-path configuration focused on cuDNN SDPA.

<a id="integrationsv2-selfforcing-tests-baselinefastvideo-readme--files-tracked-here"></a>

## Files tracked here

- `run.sh` - clone + setup + patch + benchmark runner
- `pyproject.toml` - isolated venv definition
- `changes.patch` - local edits on top of pinned FastVideo commit
- `.gitignore` - ignores local clone and venv
