# FlashDreams MIRA Mini

This integration runs the 1B MIRA Mini car-soccer world model using native
FlashDreams components. Its action encoder, diffusion transformer,
`BlockKVCache` temporal state, flow scheduler, bootstrap encoder, and causal
video decoder are implemented under `mira_integration`; the external
`alakazam-mira-mini` Python package is not a runtime dependency.

The default weights are downloaded from
[`alakazamworld/mira-mini`](https://huggingface.co/alakazamworld/mira-mini)
on the first real rollout. Config inspection does not construct the model or
download weights.

## Install

```bash
uv sync --package flashdreams-mira --extra dev --extra runners
uv pip install imageio-ffmpeg
```

`huggingface_hub` reads `HF_TOKEN` from the environment. Do not put a token in
the runner config or command line. The 1B bundle is roughly 12 GB and requires
an NVIDIA GPU. Its weights are CC BY-NC-SA 4.0 (non-commercial, share-alike).

The bootstrap video encoder uses the DINOv3 architecture loaded through
PyTorch Hub; its weights are restored from the MIRA checkpoint rather than
downloaded separately.

## Demo

Inspect the resolved config without loading a model:

```bash
uv run flashdreams-run --no-instantiate mira-mini-1b-demo
```

Generate the scripted example (forward, steer right, forward, steer left):

```bash
uv run flashdreams-run mira-mini-1b-demo
```

The output directory contains `mira-mini-1b-demo.mp4` and
`stats_mira-mini-1b-demo.json`. Override the action script using
`KEY+KEY@STEPS` segments:

```bash
uv run flashdreams-run mira-mini-1b-demo \
  --action-script 'W@10,W+D@6,Space@2,W+A@6'
```

Use local files instead of Hugging Face by overriding the pipeline paths:

```bash
uv run flashdreams-run mira-mini-1b-demo \
  --pipeline.bundle-path C:/models/mira-mini \
  --pipeline.checkpoint-path C:/models/mira-mini/checkpoint-52000/checkpoint.pth \
  --pipeline.context-path C:/models/mira-mini/context/default.npz
```

MIRA Mini generation is single-GPU. CPU execution is supported for focused
unit tests, not for the example rollout.

## Programmatic pipeline

```python
from mira_integration.config import PIPELINE_MIRA_MINI_1B

pipeline = PIPELINE_MIRA_MINI_1B.setup().to("cuda").eval()
cache = pipeline.initialize_cache(n_diffusion_steps=2)
frames = pipeline.generate(0, cache, input=["W", "D"])  # [T, C, H, W]
pipeline.finalize(0, cache)
pipeline.close()
```

## References and licenses

- Architecture: [mira-wm/mira](https://github.com/mira-wm/mira), Apache-2.0.
- Example bundle: [Alakazam Studios/alakazam-mira-mini](https://github.com/Alakazam-studios/alakazam-mira-mini), MIT.
- Weights: [alakazamworld](https://huggingface.co/alakazamworld), CC BY-NC-SA 4.0.

The Alakazam repository is a format and demo reference only; FlashDreams does
not import its package at runtime.
