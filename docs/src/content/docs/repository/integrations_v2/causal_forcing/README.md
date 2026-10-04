---
title: 'flashdreams-causal-forcing'
---

<a id="integrationsv2-causalforcing-readme--flashdreams-causal-forcing"></a>

<a id="integrationsv2-causalforcing-readme--integration-links"></a>

## Integration links

- **Applications:** [T2V](apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/causal_forcing/config.py)

Causal-Forcing chunkwise / framewise streaming T2V + I2V inference for Wan 2.1 1.3B,
packaged as a [`flashdreams`](https://github.com/NVIDIA/flashdreams) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](../../../developer_guides/new_integration.md)
developer-guide flow.

<a id="integrationsv2-causalforcing-readme--shipped-pipeline-configs"></a>

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `causal-forcing-wan2.1-t2v-1.3b-chunkwise` | Causal-Forcing chunkwise Wan 2.1 1.3B T2V (`len_t=3`). |
| `causal-forcing-wan2.1-t2v-1.3b-framewise` | Causal-Forcing framewise Wan 2.1 1.3B T2V (`len_t=1`). |
| `causal-forcing-wan2.1-i2v-1.3b-framewise` | Causal-Forcing framewise Wan 2.1 1.3B I2V (`len_t=1`). |

<a id="integrationsv2-causalforcing-readme--application-integrations"></a>

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise` | `causal-forcing-wan2.1-t2v-1.3b-chunkwise` |
| `t2v-causal-forcing-wan2.1-t2v-1.3b-framewise` | `causal-forcing-wan2.1-t2v-1.3b-framewise` |

The I2V config remains available for direct pipeline use.

<a id="integrationsv2-causalforcing-readme--install"></a>

## Install

The plugin is registered as a `uv` workspace member. From the repository root,
sync it and its dependencies with:

```bash

uv sync --package flashdreams-causal-forcing

```

<a id="integrationsv2-causalforcing-readme--hugging-face-setup"></a>

## Hugging Face setup

Public checkpoints are downloaded from Hugging Face on first use. `HF_TOKEN`
is optional for public repositories, but can be set when authentication is
needed or to avoid lower anonymous rate limits.

```bash

# huggingface token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default

```

<a id="integrationsv2-causalforcing-readme--run"></a>

## Run

```bash

uv run --package flashdreams-causal-forcing flashdreams-run-v2 \
  t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise --output-path artifacts/t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise.mp4 -- \
  --prompt "A cat surfing." --total-blocks 21 --no-compile

```

See [apps/t2v/README.md](apps/t2v/README.md) for the concise launch command
and [the shared T2V guide](../../apps/t2v/README.md) for common arguments.

<a id="integrationsv2-causalforcing-readme--direct-pipeline-access"></a>

## Direct pipeline access

The lower-level recipe pipeline can also be instantiated directly:

```python

import torch
from causal_forcing.config import PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=480 // sp, # latent height for DiT
    width=832 // sp, # latent width for DiT
)

total_blocks: int = 21
generated_chunks: list[torch.Tensor] = []
for i in range(total_blocks):
    video_chunk = pipeline.generate(autoregressive_index=i, cache=cache)
    pipeline.finalize(autoregressive_index=i, cache=cache) # advance streaming caches
    generated_chunks.append(video_chunk.cpu()) # each chunk is [T, C, H, W]

```

<a id="integrationsv2-causalforcing-readme--tests"></a>

## Tests

```bash

uv run --group test pytest integrations_v2/causal_forcing/tests -m ci_cpu
```
