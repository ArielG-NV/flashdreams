---
title: 'flashdreams-cosmos-predict2'
---

<a id="integrationsv2-cosmospredict2-readme--flashdreams-cosmos-predict2"></a>

<a id="integrationsv2-cosmospredict2-readme--integration-links"></a>

## Integration links

- **Applications:** [T2V](apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/cosmos_predict2/config.py)

Cosmos-Predict2.5 2B bidirectional T2V and I2V inference,
packaged as a [`flashdreams`](https://github.com/NVIDIA/flashdreams) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](../../../developer_guides/new_integration.md)
developer-guide flow.

Cosmos-Predict2 is **bidirectional**: it generates the complete clip in one
rollout instead of advancing through multiple causal blocks. It therefore
requires exactly one block (`--total-blocks 1`); multi-block generation is not
supported.

<a id="integrationsv2-cosmospredict2-readme--shipped-pipeline-configs"></a>

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `cosmos2-t2v-2b-720p` | Cosmos-Predict2.5 2B T2V at 720p (single AR step, prompt-only). |
| `cosmos2-i2v-2b-720p` | Cosmos-Predict2.5 2B I2V at 720p (single AR step, prompt + first-frame image). |

<a id="integrationsv2-cosmospredict2-readme--application-integrations"></a>

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-cosmos2-t2v-2b-720p` | `cosmos2-t2v-2b-720p` |

The I2V config remains available for direct pipeline use.

<a id="integrationsv2-cosmospredict2-readme--install"></a>

## Install

The plugin is registered as a `uv` workspace member. From the repository root,
sync it and its dependencies with:

```bash

uv sync --package flashdreams-cosmos-predict2

```

Standalone (outside the workspace) also works:

```bash

uv pip install -e integrations_v2/cosmos_predict2

```

<a id="integrationsv2-cosmospredict2-readme--hugging-face-setup"></a>

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

<a id="integrationsv2-cosmospredict2-readme--run"></a>

## Run

```bash

uv run --package flashdreams-cosmos-predict2 flashdreams-run-v2 \
  t2v-cosmos2-t2v-2b-720p --output-path artifacts/t2v-cosmos2-t2v-2b-720p.mp4 -- \
  --prompt "A cat surfing." --no-compile

```

See [apps/t2v/README.md](apps/t2v/README.md) for the concise launch command
and [the shared T2V guide](../../apps/t2v/README.md) for common arguments.

<a id="integrationsv2-cosmospredict2-readme--direct-pipeline-access"></a>

## Direct pipeline access

The lower-level recipe pipeline can also be instantiated directly:

```python

from cosmos_predict2.config import PIPELINE_COSMOS2_T2V_2B_720P as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=720 // sp, # latent height for DiT
    width=1280 // sp, # latent width for DiT
)

video = pipeline.generate(autoregressive_index=0, cache=cache)
pipeline.finalize(autoregressive_index=0, cache=cache) # update one-step stats

```

<a id="integrationsv2-cosmospredict2-readme--tests"></a>

## Tests

```bash

uv run --package flashdreams-cosmos-predict2 --extra dev pytest \
  integrations_v2/cosmos_predict2/tests
```
