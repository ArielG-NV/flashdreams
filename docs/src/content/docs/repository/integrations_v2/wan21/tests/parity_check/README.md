---
title: 'Wan2.1 official parity check'
---

<a id="integrationsv2-wan21-tests-paritycheck-readme--wan2-1-official-parity-check"></a>

<a id="integrationsv2-wan21-tests-paritycheck-readme--integration-links"></a>

## Integration links

- **Applications:** [wan21 applications](../../../../../models/wan21.md#developer-details)
- **Configuration:** [wan21 configuration](../../../../../models/wan21.md#developer-details)

This test runs a pinned revision of the official Wan2.1 repository with the
upstream `t2v-1.3B` command-line setup, while forcing the cuDNN SDPA backend and
`torch.compile`.

<a id="integrationsv2-wan21-tests-paritycheck-readme--requirements"></a>

## Requirements

- Bash, Git, and `uv`
- An NVIDIA GPU supported by PyTorch 2.11 or newer and the cuDNN SDPA backend
- Network access and enough local storage to clone Wan2.1, install its Python
  dependencies, and download the `Wan-AI/Wan2.1-T2V-1.3B` checkpoint

<a id="integrationsv2-wan21-tests-paritycheck-readme--what-this-harness-does"></a>

## What this harness does

1. Clones `Wan-Video/Wan2.1` and checks out a pinned commit.
2. Applies `changes.patch` (cuDNN SDPA enforcement in fallback attention path).
3. Creates a Python 3.12 `.venv`, installs PyTorch 2.11 or newer and
   Torchvision 0.26 or newer, then installs the upstream requirements without
   its Torch, Torchvision, or `flash_attn` pins.
4. Installs `einops` (missing from the pinned upstream requirements) and
   `huggingface_hub[cli]`.
5. Installs the local `flashdreams` package in editable mode with `--no-deps`
   for environment alignment.
6. Downloads `Wan-AI/Wan2.1-T2V-1.3B` if missing.
7. Runs:

```bash

python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

```

The upstream official code path already uses `tqdm` over diffusion timesteps in `wan/text2video.py`.

<a id="integrationsv2-wan21-tests-paritycheck-readme--run"></a>

## Run

```bash

bash integrations_v2/wan21/tests/parity_check/run.sh
```
