---
title: 'flashdreams-wan21'
---

<a id="integrationsv2-wan21-readme--flashdreams-wan21"></a>

<a id="integrationsv2-wan21-readme--integration-links"></a>

## Integration links

- **Applications:** [T2V](apps/t2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/wan21/config.py)

Wan 2.1 bidirectional T2V + I2V inference,
packaged as a [`flashdreams`](https://github.com/NVIDIA/flashdreams) workspace plugin.

This is a worked example of the
[Add a new method](../../../developer_guides/new_integration.md)
developer-guide flow.

Wan 2.1 is **bidirectional**: it generates the complete clip in one rollout
instead of advancing through multiple causal blocks. It therefore requires
exactly one block (`--total-blocks 1`); multi-block generation is not
supported.

<a id="integrationsv2-wan21-readme--shipped-pipeline-configs"></a>

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `wan21-t2v-1.3b-480p` | Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only). |
| `wan21-i2v-14b-480p` | Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame). |

<a id="integrationsv2-wan21-readme--application-integrations"></a>

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-wan21-t2v-1.3b-480p` | `wan21-t2v-1.3b-480p` |

The I2V config remains available for direct pipeline use.

<a id="integrationsv2-wan21-readme--install"></a>

## Install

The plugin is registered as a `uv` workspace member. From the repository root,
sync it and its dependencies with:

```bash

uv sync --package flashdreams-wan21

```

<a id="integrationsv2-wan21-readme--hugging-face-setup"></a>

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

<a id="integrationsv2-wan21-readme--run"></a>

## Run

Generate the model's single-block MP4:

```bash

uv run --package flashdreams-wan21 flashdreams-run-v2 \
  t2v-wan21-t2v-1.3b-480p --output-path artifacts/t2v-wan21-t2v-1.3b-480p.mp4 -- \
  --prompt "A cat surfing." --no-compile

```

See the [application README](apps/t2v/README.md) for the concise launch command
and the [shared T2V guide](../../apps/t2v/README.md) for common arguments.

<a id="integrationsv2-wan21-readme--direct-pipeline-access"></a>

## Direct pipeline access

The lower-level recipe pipeline can also be instantiated directly:

```python

from wan21.config import PIPELINE_WAN21_T2V_1PT3B_480P as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=480 // sp, # latent height for DiT
    width=832 // sp, # latent width for DiT
)

video = pipeline.generate(autoregressive_index=0, cache=cache)
pipeline.finalize(autoregressive_index=0, cache=cache) # update one-step stats

```

<a id="integrationsv2-wan21-readme--tests"></a>

## Tests

```bash

uv run --group test pytest integrations_v2/wan21/tests -m ci_cpu
```
