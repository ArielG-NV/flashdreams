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
  --demo mira-mini-1-player-1b-8-step \
  --action-script 'W+D@5,W+A@5,Space@6,W@5'
```

Each action-script suffix is a duration in 100 ms units, so `W@3` holds `W`
for 300 ms. The runner converts that duration to generated chunks using the
configured `fps` and the selected demo's `frames_per_chunk`.

The runner writes `mira.mp4`, `stats_mira.json`, and `metrics_mira.csv` under
`artifacts/mira/<mira-demo-name>/`. The CSV report summarizes model metrics.
Average FPS uses monotonic frame-request and media-push completion timestamps.
The cumulative completed-frame number is carried through the same timing
record, so the runner does not enable synchronized pipeline profiling.

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
  --demo mira-mini-4-player-1b-8-step \
  --host 0.0.0.0 --port 8083

# launch 1 player mira demo
uv run mira-webrtc \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-1-player-1b-8-step \
  --host 0.0.0.0 --port 8083

  # launch 1 player 364m mira demo
uv run mira-webrtc \
  --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
  --demo mira-mini-1-player-364m-8-step \
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
    --demo mira-mini-1-player-1b-8-step \
    --action-script 'W+D@5,W+A@5,Space@6,W@5'
```

## Quality Evaluation

Render and evaluate every demo in the packaged car-soccer manifest with:

```bash
uv run flashdreams-run calculate-mira-quality \
  --action-script 'W+D@5,W+A@5,Space@6,W@5'
```

The command renders each concrete demo multiple times using the supplied action
script and consecutive seeds; These values are used to compute quality metrics:
* `temporal_instability_metric`: average motion-compensated pixel-boiling score
    * Stored in `artifacts/mira/temporal-instability/`

## Results Viewer

Combine metrics from any number of result folders and open a pandas-generated
report after running the MIRA demo and Quality Evaluation suite:

```bash
uv run flashdreams-run mira-results-viewer \
  mira_metrics_folder_1 mira_metrics_folder_2 ... \
  --temporal-instability-mira-folder <mira_folder> \
  --ignore-runner-slug <runner_slug_to_skip>

# Example:
uv run flashdreams-run mira-results-viewer artifacts/mira/ \
  --temporal-instability-mira-folder artifacts/mira/ \
  --ignore-runner-slug mira-mini-4-player-1b-8-step
```

For each supplied folder, the viewer reads every direct child matching
`<metrics-folder>/<slug>/metrics_mira.csv`. The temporal-instability folder is
required. It writes `runner_gpu_quality.csv` and an HTML report with one table
row per runner and one average-FPS column per discovered GPU. Separately, it
writes one `pareto_curve_mira_mini_<gpu>.svg` per GPU. Each SVG contains only
that GPU's non-dominated average-FPS/quality points and defines the superscripted
Quality metric below the curve; the SVGs are not embedded in the HTML report.

Pass one or more names after `--ignore-runner-slug` to remove
runner slugs before aggregation.

The HTML report color-codes FPS below 15 red,
15 through 30 yellow, and above 30 green, and records the source repository
commit below the table. Rows are ordered by relative quality, with 100% first.
Quality is temporal stability expressed using relative distance; table values
also show the underlying temporal-instability metric in parentheses.

## Comparison Chart Generator

Generate a focused comparison from one MIRA result folder:

```bash
uv run flashdreams-run mira-comparison-chart-generator artifacts/mira/ \
  --metric-to-compare runtime_average_fps \
  --runner-slug-direct-compare mira-mini-1-player-1b-8-step \
  --flashdreams-gpu-to-compare-with "NVIDIA RTX PRO 6000 Blackwell.*" \
  --competitor-gpu-to-compare-with "NVIDIA RTX PRO 6000 Blackwell" \
  --custom-y-axis "Average FPS" \
  --custom-title "MIRA Mini Comparison [Workstation]" \
  --target-file-name "mira-mini-throughput.svg" \
  --flashdreams-gpu-other-runner mira-mini-1-player-1b-4-step \
  --flashdreams-gpu-other-runner mira-mini-1-player-364m-4-step
```

The Competitor bar is red and leftmost. The matching direct FlashDreams bar
and any repeated `--flashdreams-gpu-other-runner` bars are green and appear to
its right. Both `--custom-y-axis` and `--custom-title` are required. The
Competitor legend entry carries an Oxford-style superscript citation to the
source URL printed below the chart. The chart, stacked legend, and citation use
relative layout rows, and the figure height grows with rotated runner-label
length so content is not clipped. `--target-file-name` is required, accepts
only a bare `.svg` filename, and writes it below
`artifacts/mira-comparison-chart/`.

## Troubleshooting

> Automatically "disconnecting from preview", unable to view preview or join a session
Likely due to browser or local network security. Try disabling WebRTC local IP hiding.
Example: for firefox, change in `about:config`, `media.peerconnection.ice.obfuscate_host_addresses` to `false`.
