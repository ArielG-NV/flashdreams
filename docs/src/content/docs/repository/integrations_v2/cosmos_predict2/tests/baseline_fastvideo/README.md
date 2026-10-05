---
title: 'FastVideo Cosmos 2.5 baseline'
---

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--fastvideo-cosmos-2-5-baseline"></a>

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--integration-links"></a>

## Integration links

- **Applications:** [cosmos predict2 applications](../../../../../models/cosmos_predict2.md#developer-details)
- **Configuration:** [cosmos predict2 configuration](../../../../../models/cosmos_predict2.md#developer-details)

Self-contained baseline run of upstream [FastVideo](https://github.com/hao-ai-lab/FastVideo)
for the Cosmos 2.5 text-to-world (T2W) path, aligned with flashdreams parity
conventions.

This harness:

- clones upstream FastVideo at a pinned commit,
- applies `changes.patch`,
- uses an isolated `uv` env with FastVideo-compatible deps, then installs local
  `flashdreams` (no-deps) to keep the parity env aligned with the main workspace,
- runs FastVideo's Cosmos generation example with parity-oriented settings.

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--requirements"></a>

## Requirements

- Linux or WSL with Bash, Git, and `uv`
- one CUDA-capable NVIDIA GPU with enough memory for the 2B model at 720p
- network access on the first run to clone FastVideo and download
  `KyleShao/Cosmos-Predict2.5-2B-Diffusers`

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--run"></a>

## Run

From this directory:

```bash

bash run.sh

```

The script reuses an existing clone and applied patch. `uv sync` reconciles the
isolated environment on every run.

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--outputs"></a>

## Outputs

Written under `FastVideo/`:

- `outputs_video/cosmos2_5_t2w.mp4` - generated video

This harness does not currently emit timing JSON. `run.sh` exports
`FASTVIDEO_PARITY_STATS_PATH`, but the pinned FastVideo commit and local patch do
not consume it.

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--backend-and-speed-settings"></a>

## Backend and speed settings

The baseline runs with:
- CPU offload disabled (`dit/text_encoder/vae` offload all `False`)
- `enable_torch_compile=True`
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- `FASTVIDEO_FORCE_CUDNN_SDPA=1`

This enforces a fast-path configuration and applies strict cuDNN SDPA forcing
inside FastVideo's SDPA backend when the env flag is set.

`flashdreams` is installed with `uv pip install --no-deps -e ...` for environment
alignment, without forcing flashdreams' full dependency set into this
FastVideo/Cosmos parity environment.

<a id="integrationsv2-cosmospredict2-tests-baselinefastvideo-readme--files-tracked-here"></a>

## Files tracked here

- `run.sh` - clone + setup + patch + baseline runner
- `pyproject.toml` - isolated venv definition
- `changes.patch` - local edits on top of pinned FastVideo commit
- `.gitignore` - ignores local clone and venv
