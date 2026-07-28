# FlashDreams MIRA Mini

This integration runs the 1B and 364M MIRA Mini car-soccer world models using
native FlashDreams components. Its action encoder, diffusion transformer,
`BlockKVCache` temporal state, flow scheduler, bootstrap encoder, and causal video decoder are implemented under `mira_integration`.

## Install

```bash
uv sync --package flashdreams-mira --extra dev --extra runners
```

`huggingface_hub` reads `HF_TOKEN` from the environment. Ensure the token exists and has premissions to access DINOv3.

## Headless Demo

Run the demo with a custom action sequence. Generated videos and timing data
are written to `artifacts/mira/<mira-demo-name>/` by default:

```bash
# W for 1.0s, then W+D for 0.6s, then Space for 0.2s, then W+A for 0.6s
LOGURU_LEVEL="WARNING" uv run flashdreams-run mira \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-1p-1b-high \
  --action-script 'W+D@5,W+A@5,Space@6,W@5'
```

Each action-script suffix is a duration in 100 ms units, so `W@3` holds `W`
for 300 ms. The runner converts that duration to generated chunks using the
configured `fps` and the selected demo's `frames_per_chunk`.

The runner writes `mira.mp4`, `stats_mira.json`, and `metrics_mira.csv` under
`artifacts/mira/<mira-demo-name>/`. The CSV report summarizes model metrics.

Use the runner-only `--demo all` sentinel to execute every demo declared in
the manifest sequentially:

```bash
LOGURU_LEVEL="WARNING" uv run flashdreams-run mira \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo all \
  --action-script 'W+D@5,W+A@5,Space@6,W@5'
```

For multiplayer demos, the action script controls player 1 and leaves the
remaining players inactive. The output MP4 tiles all configured player views.

## WebRTC Demo

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
  --demo mira-mini-1p-1b-high \
  --host 0.0.0.0 --port 8083

  # launch 1 player 364m mira demo
uv run mira-webrtc \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-1p-364m-high \
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

selection = load_demo_config("path/to/manifest.yaml", "mira-mini-1b")
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

Use Nsight Systems to profile, launch in an admin terminal (assuming git-bash):

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
    --demo mira-mini-1p-1b-high \
    --action-script 'W+D@5,W+A@5,Space@6,W@5'
```

## Quality Evaluation

After generating every demo in the packaged car-soccer manifest (i.e. use slug `--demo all` in the headless-demo), calculate quality of the generated videos with:

```bash
uv run flashdreams-run calculate-mira-quality
```

The command first requires an MP4 under every
`artifacts/mira/<demo-slug>/` directory. It then writes under `artifacts/mira/**` additional metrics in a `result.csv` per quality evaluation:
* `temporal_instability_metric`: motion-compensated pixel-boiling score
* ...

## Results Viewer

Combine metrics from any number of result folders and open a pandas-generated
report AFTER running `mira` with the `--demo all` slug:

```bash
uv run flashdreams-run mira-results-viewer \
  mira_metrics_folder_1 mira_metrics_folder_2 ... \
  --temporal-instability-mira-folder <mira_folder>

# Example:
uv run flashdreams-run mira-results-viewer artifacts/mira/ \
  --temporal-instability-mira-folder artifacts/mira/
```

For each supplied folder, the viewer reads every direct child matching
`<metrics-folder>/<slug>/metrics_mira.csv`. It writes the concatenated CSV,
an HTML table, and bar charts for average FPS, 90th-percentile FPS, model
VRAM footprint, and temporal instability under
`artifacts/mira-results-viewer/`, then opens `mira_results.html` in the default
web browser. Pandas is a Python data-analysis library rather than a native
spreadsheet application, so the terminal prints the exact local file path and
`file:` URL rendered by the browser.

## Troubleshooting

> Automatically "disconnecting from preview", unable to view preview or join a session
Likely due to browser or local network security. Try disabling WebRTC local IP hiding.
Example: for firefox, change in `about:config`, `media.peerconnection.ice.obfuscate_host_addresses` to `false`.
