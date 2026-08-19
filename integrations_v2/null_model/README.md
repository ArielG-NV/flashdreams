<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams NULL Model
## Observable contract

For autoregressive step `k`, the model accepts one scalar tensor and emits a
one-pixel RGB video chunk:

| Property | Value |
| --- | --- |
| Input | Tensor with shape `[1, 1]` |
| Output shape | Tensor with shape `[1, 3, 1, 1, 1]` |
| Output value | Output is `Input + cache.autoregressive_index` |
| Output layout | `VideoTensorLayout.bcthw` |

##Files
config.py           Defines the 'null model' pipeline
encoder.py          Encodes input into an encoded-input
transformer.py      Processes the encoded-input into a latent/flow that scheduler must 'denoise'
decoder.py          Decodes the output of the transformer

```python
NULL_MODEL_CONFIG = NullModelConfig(
    name="null-model",
    encoder=NullInputEncoderConfig(), # Encoder of inputs
    diffusion_model=DiffusionModelConfig( # "Archtype" of our pipeline
        transformer=NullTransformerConfig(), # Transformer
        scheduler=FlowMatchSchedulerConfig( # Scheduler
            num_inference_steps=1,
            denoising_timesteps=[1000],
        ),
    ),
    decoder=NullDecoderConfig(), # Decoder of latents/output-tensor
)
```

## How the integration was designed, step by step

### Make a real integration package

[`pyproject.toml`](pyproject.toml) declares `flashdreams-null-model` as a
workspace package that depends on `flashdreams`. This issolated `pyproject.toml`
This keeps our integration isolated so it can safely declare dependencies without
affecting other integrations.

### Implement the per-step encoder

[`encoder.py`](null_model/encoder.py) defines `NullInputEncoder` as a
`StreamingEncoder` (bound in config to `NullModelConfig::encoder`).
This is important because it allows us to define an encoder that runs for each auto-regressive step.
This contrasts the `NullModelConfig::diffusion_model::transformer::context_encoder` which is designed to run only once at the beginning of a session for generation.

Since this is a null encoder we just expect a simple 1 by 1 tensor and emit the same tensor.

### Implement the transformer

In order the following were defined: 
1. `latent_shape` declares one batch, three channels, one frame, and one pixel as the shape of the output tensor. 
2. `initialize_autoregressive_cache()` creates a cache object which tracks the autoregressive step of each continuous generation.
3. `initial_noise()` returns zeros so that the scheduler does not have any 'noise' to 'denoise' from the `predict_flow` method result.
4. `predict_flow()` computes the flow utilizing our 'encoded-input' tensor and the autoregressive step reported by the autoregressive cache.

The flow is computed as follows:
```text
# Explicit computation of our implemented `NullTransformer`
target = scalar_input + cache.autoregressive_index
flow   = noisy_latent - target

# `FlowMatchScheduler` implicit logic to denoise the flow into the final output tensor
clean  = noisy_latent - sigma * flow
       = noisy_latent - (noisy_latent - target)
       = target

# Result:
target == clean == scalar_input + cache.autoregressive_index
```

This is why one scheduler step is sufficient and why every output element is
exactly the expected value

### Implement the per-step decoder

[`decoder.py`](null_model/decoder.py) defines an identity `NullDecoder`. The
transformer already emits the final tensor, so there is nothing to
decode. The component still exists because it demonstrates where a real
integration would place a latent-to-pixel decoder.