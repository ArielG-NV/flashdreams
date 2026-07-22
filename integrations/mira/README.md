# FlashDreams MIRA Mini

This integration runs the 1B MIRA Mini car-soccer world model using native
FlashDreams components. Its action encoder, diffusion transformer,
`BlockKVCache` temporal state, flow scheduler, bootstrap encoder, and causal video decoder are implemented under `mira_integration`.

## Install

```bash
uv sync --package flashdreams-mira --extra dev --extra runners
uv pip install imageio-ffmpeg
```

`huggingface_hub` reads `HF_TOKEN` from the environment. Ensure the token exists and has premissions to access DINOv3.

## Demo

Run the demo with a custom action sequence. Generated videos and timing data are written to `artifacts/mira/` by default:

```bash
# W for 1.0s, then W+D for 0.6s, then Space for 0.2s, then W+A for 0.6s
uv run flashdreams-run mira \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-1p \
  --action-script 'W@5,W+D@5,Space@6,W+A@5'
```

Each action-script suffix is a duration in 100 ms units, so `W@3` holds `W`
for 300 ms. The runner converts that duration to generated chunks using the
configured `fps` and the selected demo's `frames_per_chunk`.

For multiplayer demos, the action script controls player 1 and leaves the
remaining players inactive. The output MP4 tiles all configured player views.

Launch the browser UI to host MIRA. Browser will print the `<IP>/request_session` URL to join the play session through:

```bash
# launch 4 player mira demo
uv run mira-webrtc \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-4p \
  --host 0.0.0.0 --port 8083

# launch 1 player mira demo
uv run mira-webrtc \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-1p \
  --host 0.0.0.0 --port 8083
```

## Adding new demos/checkpoints

Model definitions live in
`mira_integration/configs/*`. The manifest describes an input-map, checkpoint,
player count, sampler step count, latent grid, output frame shape, and other
metadata needed for MIRA to run.

The manifests in this folder are your `<manifest_slug>` for argument `--manifest <manifest_slug>` and the members of `demos` inside the manifest are your argument for the `--demo <demo_slug>` parameter.

## Programmatic pipeline

```python
from mira_integration.config import load_demo_config

selection = load_demo_config("path/to/manifest.yaml", "mira-mini-1p")
pipeline = selection.pipeline.setup().to("cuda").eval()
cache = pipeline.initialize_cache(n_diffusion_steps=selection.metadata.steps)
frames = pipeline.generate(0, cache, input=["W", "D"])
pipeline.finalize(0, cache)
pipeline.close()
```

## References and licenses

- Architecture: [mira-wm/mira](https://github.com/mira-wm/mira), Apache-2.0.
- Example bundle: [Alakazam Studios/alakazam-mira-mini](https://github.com/Alakazam-studios/alakazam-mira-mini), MIT.
- Weights: [alakazamworld](https://huggingface.co/alakazamworld), CC BY-NC-SA 4.0.

The Alakazam repository is a format and demo reference only; FlashDreams does
not import its package at runtime.

## Profile with NSight

Use Nsight Systems to profile:

```bash
mkdir -p artifacts/mira/nsight

nsys profile \
  --force-overwrite=true \
  --stats=true \
  --output artifacts/mira/nsight/mira-demo-%h-%p \
  --trace=cuda,cuda,nvtx,cublas-verbose,cuDNN-verbose,cusparse-verbose,nvvideo,cublas,nvtx,python-gil \
  --pytorch=functions-trace-shapes,autograd-nvtx \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=200 \
  --cuda-memory-usage=true \
  --cuda-trace-all-apis=true \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-frequency=1000 \
  uv run flashdreams-run mira \
    --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
    --demo mira-mini-1p \
    --action-script 'W@5,W+D@5,Space@6,W+A@5'
```
