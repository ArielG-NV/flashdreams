---
title: 'FlashDreams NULL Model'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-nullmodel-readme--flashdreams-null-model"></a>

<a id="integrationsv2-nullmodel-readme--integration-links"></a>

## Integration links

- **Applications:** [Colour Fade](../color_fade/README.md) and [Red Screen](../red_screen/README.md)
- **Configuration:** [`null_model/config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/config.py)

A simple pipeline with an encoder, transformer, scheduler and decoder, whose output
is arithmetic rather than video, so it needs no checkpoint and runs on a CPU. It
was written alongside the v2 protocols as the pipeline they were developed against,
something whole that could be driven end to end while they were still changing,
which is why it sits here rather than under `integrations/`.

`flashdreams/test_v2/test_client_window.py` still drives it that way, through
`IClientWindow`, `InputSource` and `OutputSink`. It implements none of those
itself and registers no entry point, so `flashdreams-run-v2` cannot reach it. For
a v2 application to copy, use [color_fade](../color_fade/README.md) for the
file path or [red_screen](../red_screen/README.md) for the interactive one, and
read the [integration guide](../README.md). What follows is about the pipeline.

<a id="integrationsv2-nullmodel-readme--observable-contract"></a>

## Observable contract

| Property | Value |
| --- | --- |
| Input | Tensor with shape `[1, 1]` |
| Output shape | Tensor with shape `[1, 3, 1, 1, 1]` |
| Output value | `Input + cache.autoregressive_index` |
| Output layout | `VideoTensorLayout.bcthw` |

<a id="integrationsv2-nullmodel-readme--files"></a>

## Files

| File | What it does |
| --- | --- |
| `config.py` | Defines the null-model pipeline. |
| `encoder.py` | Adds 100 to the input, as minor obfuscation. |
| `transformer.py` | Turns the encoded input into a flow for the scheduler to denoise. |
| `decoder.py` | Subtracts the 100 back off. |

```python

NULL_MODEL_CONFIG = NullModelConfig(
    name="null-model",
    encoder=NullInputEncoderConfig(),
    diffusion_model=DiffusionModelConfig(
        transformer=NullTransformerConfig(),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=1,
            denoising_timesteps=[1000],
        ),
    ),
    decoder=NullDecoderConfig(),
)

```

<a id="integrationsv2-nullmodel-readme--how-the-pipeline-is-put-together"></a>

## How the pipeline is put together

<a id="integrationsv2-nullmodel-readme--a-real-integration-package"></a>

### A real integration package

[`pyproject.toml`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/pyproject.toml) declares `flashdreams-null-model` as a
workspace package depending on `flashdreams`.

<a id="integrationsv2-nullmodel-readme--the-per-step-encoder"></a>

### The per-step encoder

[`encoder.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/encoder.py) defines `NullInputEncoder` as a
`StreamingEncoder`, bound in the config to `NullModelConfig.encoder`. A streaming
encoder runs on every autoregressive step, unlike
`NullModelConfig.diffusion_model.transformer.context_encoder`, which runs once at
the start of a generation.

It adds 100 to the 1x1 input tensor.

<a id="integrationsv2-nullmodel-readme--the-transformer"></a>

### The transformer

[`transformer.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/transformer.py) defines, in order:

1. `latent_shape` — one batch, three channels, one frame, one pixel.
2. `initialize_autoregressive_cache()` — a cache tracking the autoregressive step
   of a continuous generation.
3. `initial_noise()` — zeros, so the scheduler has no noise to denoise beyond the
   flow `predict_flow` returns.
4. `predict_flow()` — the flow, from the encoded input and the step index the
   cache reports.

```text

# NullTransformer
target = encoded_input + cache.autoregressive_index
flow   = noisy_latent - target

# FlowMatchScheduler, sigma 1.0, one step
clean = noisy_latent - 1.0 * flow
      = noisy_latent - (noisy_latent - target)
      = target

```

Which is why one scheduler step is enough, and why every output tensor is exactly
the expected value.

<a id="integrationsv2-nullmodel-readme--the-per-step-decoder"></a>

### The per-step decoder

[`decoder.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/decoder.py) defines `NullDecoder`, which subtracts the
encoder's 100 back off the transformer's output.

<a id="integrationsv2-nullmodel-readme--tests"></a>

## Tests

```bash

uv run --no-sync pytest integrations_v2/null_model -m ci_cpu
```
