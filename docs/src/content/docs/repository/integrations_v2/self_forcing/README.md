---
title: 'flashdreams-self-forcing'
---

<a id="integrationsv2-selfforcing-readme--flashdreams-self-forcing"></a>

<a id="integrationsv2-selfforcing-readme--integration-links"></a>

## Integration links

- **Applications:** [T2V](apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/self_forcing/config.py)

Self-Forcing distilled streaming T2V inference for Wan 2.1 1.3B,
packaged as a [`flashdreams`](https://github.com/NVIDIA/flashdreams) integration plugin.

This is a worked example of the
[Add a new method](../../../developer_guides/new_integration.md)
developer-guide flow.

<a id="integrationsv2-selfforcing-readme--shipped-pipeline-configs"></a>

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `self-forcing-wan2.1-t2v-1.3b` | Self-Forcing distilled Wan 2.1 1.3B T2V (Wan VAE decoder, 4-step). |
| `self-forcing-wan2.1-t2v-1.3b-taehv` | Same DiT, swapped to the TAEHV (LightTAE) decoder for faster decoding. |
| `self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` | Long-rollout preset with static sink=5 + window=7 + KVCache-relative RoPE. |

<a id="integrationsv2-selfforcing-readme--application-integrations"></a>

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-self-forcing-wan2.1-t2v-1.3b` | `self-forcing-wan2.1-t2v-1.3b` |
| `t2v-self-forcing-wan2.1-t2v-1.3b-taehv` | `self-forcing-wan2.1-t2v-1.3b-taehv` |
| `t2v-self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` | `self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` |

<a id="integrationsv2-selfforcing-readme--install"></a>

## Install

From the repository root, sync the Self-Forcing workspace package and its
dependencies:

```bash

uv sync --project integrations_v2/self_forcing

```

For an editable install into the active environment instead:

```bash

uv pip install -e integrations_v2/self_forcing

```

<a id="integrationsv2-selfforcing-readme--hugging-face-setup"></a>

## Hugging Face setup

Public checkpoints are downloaded from Hugging Face on first use. No token is
required for the public Self-Forcing and Wan repositories. Set `HF_TOKEN` only
when your Hugging Face access or rate limits require authentication.

```bash

# Optional Hugging Face token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default

```

<a id="integrationsv2-selfforcing-readme--run"></a>

## Run

```bash

uv run --package flashdreams-self-forcing flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b --output-path artifacts/t2v-self-forcing-wan2.1-t2v-1.3b.mp4 -- \
  --prompt "A cat surfing." --total-blocks 7 --no-compile

```

See [apps/t2v/README.md](apps/t2v/README.md) for the concise launch command
and [the shared T2V guide](../../apps/t2v/README.md) for common arguments.

<a id="integrationsv2-selfforcing-readme--low-level-pipeline-access"></a>

## Low-level pipeline access

The model pipeline can also be driven directly:

```python

import torch
from self_forcing.config import PIPELINE_WAN21_T2V_1PT3B as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"],  # set a new prompt
    height=480 // sp,  # latent height for DiT
    width=832 // sp,  # latent width for DiT
)

total_blocks: int = 7
generated_chunks: list[torch.Tensor] = []
for i in range(total_blocks):
    video_chunk = pipeline.generate(autoregressive_index=i, cache=cache)
    pipeline.finalize(autoregressive_index=i, cache=cache)  # update KV cache
    generated_chunks.append(video_chunk.cpu())  # each chunk is [T, C, H, W]

```

<a id="integrationsv2-selfforcing-readme--tests"></a>

## Tests

```bash

uv run --project integrations_v2/self_forcing --extra dev \
  pytest integrations_v2/self_forcing/tests -m ci_cpu
```
