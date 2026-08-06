<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# `omnidreams`

Omnidreams integration package for `flashdreams`.

## Hugging Face assets

Omnidreams resolves public Omni Dreams assets from the `nvidia` Hugging Face
org:

- `nvidia/omni-dreams-models` for checkpoints.
- `nvidia/omni-dreams-samples` for bundled example data.
- `nvidia/omni-dreams-scenes` for WebRTC scenes.

Set `HF_TOKEN` to a token with access to these repos before running or
importing FlashDreams:

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>
```

## Installing dependencies

```bash
# Install cuda toolkit
wget wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/<arch>/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13
```

## Run batch evaluation

The `omnidreams-eval` CLI automates a fixed-split evaluation flow for
OmniDreams scene batches:

1. Discover Hugging Face scene assets and write a JSONL manifest.
2. Plan byte- or count-capped batches.
3. Stage one batch into local scratch storage.
4. Run FlashDreams generation for the staged cases.
5. Validate generated artifacts and runner logs.
6. Stage/run DrivingGen FVD-lite and WorldLens consistency evaluators.
7. Write a JSON and Markdown summary report.
8. Optionally compare the summary against a checked-in metric baseline.

The high-level workflow is:

```bash
RUN=/trees/$USER/od-runs/od-26.01
SCRATCH=/local_nvme/$USER/omnidreams-eval-scratch
BASELINE=integrations/omnidreams/eval_baselines/od-26.01-worldlens-40-v1.json

uv run --package flashdreams-omnidreams omnidreams-eval discover \
  --output "$RUN/manifest.jsonl"

uv run --package flashdreams-omnidreams omnidreams-eval plan-batches \
  --manifest "$RUN/manifest.jsonl" \
  --output "$RUN/batches.json" \
  --batch-size 20

uv run --package flashdreams-omnidreams omnidreams-eval stage-batch \
  --manifest "$RUN/manifest.jsonl" \
  --batch-plan "$RUN/batches.json" \
  --batch-id batch-00000 \
  --scratch-root "$SCRATCH" \
  --output "$RUN/staged/batch-00000.jsonl"

uv run --package flashdreams-omnidreams omnidreams-eval generate \
  --staged-manifest "$RUN/staged/batch-00000.jsonl" \
  --run-root "$RUN"

uv run --package flashdreams-omnidreams omnidreams-eval validate-generation \
  --run-root "$RUN" \
  --output "$RUN/validation.json"

uv run --package flashdreams-omnidreams omnidreams-eval summarize-run \
  --run-root "$RUN"

uv run --package flashdreams-omnidreams omnidreams-eval check-baseline \
  --summary "$RUN/evaluation-summary.json" \
  --baseline "$BASELINE" \
  --output-json "$RUN/baseline-check.json"
```

External evaluator setup is intentionally separate from FlashDreams generation,
because DrivingGen and WorldLens have their own dependencies and checkpoint
caches. Use `setup-evaluator` for DrivingGen and `setup-worldlens` for
WorldLens, then run the corresponding `prepare-*` and evaluator commands. The
adapter modules pin the upstream GitHub URLs and revisions used today; moving
those pins into shared config or a maintained fork is a reasonable follow-up if
the evaluator stack becomes long-lived.

Runtime depends mostly on FlashDreams generation and evaluator environment
setup. On a workstation-class GPU such as an RTX 6000 Pro, 20-scene batches are
intended to be practical, while the full Hugging Face scene set should be run in
batches to avoid staging all 1-2 GB scenes at once. Evaluator setup can also
download model checkpoints and may take several minutes on first use.

Interpret the report as follows:

- Validation checks generation completeness, frame counts, runner schedules,
  and missing artifacts. Any validation failure should be inspected before
  trusting evaluator metrics.
- DrivingGen FVD-lite is a regression metric. Lower is better only when
  comparing the same fixed scene split across model versions. Do not compare
  `batch-00000` directly against `batch-00001` as a quality claim.
- DrivingGen reference-vs-reference FVD is diagnostic only; it measures split
  diversity, not OmniDreams quality.
- WorldLens temporal and subject consistency are roughly higher-is-better, with
  1.0 as an idealized upper bound. They are useful standalone video-consistency
  signals, but they do not directly measure closed-loop simulator quality,
  path correctness, off-road behavior, or collisions.
- `check-baseline` compares a run summary against a JSON file containing the
  accepted metric envelope. Keep generated clips as run artifacts, not in the
  baseline JSON; the baseline should contain only expected metric values and
  tolerances.

## Run interactive-drive (desktop demo)
The `omnidreams.interactive_drive` subpackage ships a single-process
driving demo. Refer to [README for full guide](omnidreams/interactive_drive/README.md)

Example execution below:
```bash
# Token For Asset Repos
export HF_TOKEN=<YOUR-HF-TOKEN>
# Enable long paths to avoid breaking third-party source checkouts
git config --system core.longpaths true

# Sync dependencies
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams python integrations/omnidreams/omnidreams_singleview/tools/sync_thirdparty.py sync

# Prepare to run tuned for performance
uv run --package flashdreams-omnidreams omnidreams-prepare --perf
# Run demo
uv run --package flashdreams-omnidreams interactive-drive \
	--manifest example_world_model_perf.yaml --auto-start --game-mode

# add `--stream-mjpeg :8080` to stream to your browser
```

## Native DiT defaults

NVIDIA OmniDreams native DiT acceleration remains gated by the pipeline config's
`native_dit_acceleration` policy (`disabled`, `auto`, or `required`). When that
native path is enabled, the default compute profile is the FP8 KV-cache backend
with cuDNN attention:

- `native_dit_backend="fp8_kvcache_cudnn"`
- `native_dit_attention_backend="auto"` (currently resolves to cuDNN)

Set `native_dit_attention_backend="sparge"`, `"sage3"`, or `"sage3_fp8"`
explicitly to opt into Sparge/SageAttention-3 experiments. Use
`native_dit_sparge_hybrid_period > 1` with `"sparge"` to enable the FP8
Sparge/SageAttention-3 hybrid schedule when the extension and GPU support it.

## Run WebRTC server

From the workspace root, run:

```bash
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 \
  -m omnidreams.webrtc.server \
  --manifest example_world_model_perf.yaml \
  --scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4 \
  --port 8089 --player-count 2 \
  --single-gpu-multiplayer

# Recommended for two simultaneous drivers when two GPUs are available.
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 \
  -m omnidreams.webrtc.server \
  --manifest example_world_model_perf.yaml \
  --scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4 \
  --port 8089 --player-count 2 --player-devices cuda:0,cuda:1
```

When `--scene_dir` is omitted, the server downloads the selected scene from the
configured Hugging Face org, extracts its `clipgt-<uuid>[-<variant>].usdz`
archive, and stages it under `FLASHDREAMS_CACHE_DIR` (or `~/.cache/flashdreams`).
If `--scene-uuid` is omitted too, the server uses the default WebRTC scene.
Weather variants ship as sibling archives; pass `--scene-variant rain` (or
`snow`) to serve one (default is the clear-weather scene). The runtime seeds
from the scene's first ground-truth camera frame
(`clipgt/frames/<camera>/<ts>.jpeg`, falling back to `clipgt/first_image.*`) and
the weather-matched `clipgt/prompt<N>.txt` (falling back to `clipgt/prompt.txt`).
Pass `--scene_dir <path>` to use a pre-staged local scene instead.

`-player-count` (also spelled `--player-count`) accepts a positive integer and
defaults to one. Open `/game-manager` to see every player perspective, claim an
available car, inspect the shared top-down map, and review the controls. Player
claims are performed atomically with WebRTC offer creation; one car cannot be
controlled by two browser sessions.

Every player owns a full stateful OmniDreams pipeline and autoregressive cache.
Players assigned to the same GPU therefore serialize model steps. The default
lobby serves one cached scene snapshot per player and never starts background
preview inference. Browser thumbnails are not polled, metadata/BEV updates run
every two seconds only while the lobby is visible, and the hidden drive canvas
does not animate. For simultaneous drivers, use `--player-devices` with one
distinct GPU per player. Passing `torchrun --nproc_per_node 1` initializes the
server process but does not make multiple players run in parallel on one GPU.
Use `--live-lobby-previews` (or the legacy alias
`--keep-lobby-previews-active`) only when moving idle-player thumbnails are more
important than inference headroom.

The periodic `WebRTC perf` log separates `input_wait_ms`, `model_ms`,
`inference_wait_ms`, `physics_wait_ms`, `enqueue_ms`, queue depth, and
control latency. Sustained `inference_wait_ms` indicates same-device player
contention; queue depth and `enqueue_ms` indicate video-delivery backpressure.

The opt-in `--single-gpu-multiplayer` preset changes the untouched
`1280x704` default to `896x496` and enables eager control chunks. Eager
sampling starts model work from the latest held control instead of first waiting
for the 8-frame input window to close. A direction or key-release event that
arrives after a chunk has been sampled applies to the following chunk. Omit the
flag to retain full resolution and window-complete input sampling, or combine
`--eager-control-chunks` with an explicit tested resolution to tune the two
controls independently.

Multiplayer advances every player pipeline through one synchronized chunk
barrier, including neutral-input players without an active browser. On one GPU,
those stateful model passes serialize.

One measured GB300 run on 2026-08-06 used the perf manifest, two synchronized
players, `896x496`, and 20 warmed 8-frame chunks. It delivered 14.18 FPS per
player view with 563 ms median and 585 ms p90 chunk latency. The benchmark
process reached 105.3 GB GPU memory. Cold startup for both runtimes was about
607 seconds and the first compile-heavy synchronized chunk took 88.4 seconds.
The steady server log attributed about 225 ms median to one player's model pass,
226 ms to same-device inference waiting, and 10 ms to shared physics. This is a
functional one-GPU fallback, not a 30 FPS real-time configuration; use distinct
entries in `--player-devices` when lower synchronized latency is required.

To enable video post-processing by default, pass a registered preset such as
`--postprocess-preset rtx-super-resolution`. RTX postprocess presets require the
optional NVIDIA VFX runtime:

```bash
uv sync --package flashdreams-omnidreams --extra rtx-postprocess
```

The request-session page only offers a **Post-process** selector when the server
was launched with `--postprocess-preset`; the selector can toggle that launched
preset off for the next connection.

## Run gRPC server

From the workspace root, run:

```bash
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 -m omnidreams.grpc.server --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf --host 0.0.0.0 --port 50051
```

The server implements `omnidreams.grpc.protos.video_model.WorldModelService`
and listens on `0.0.0.0:50051` by default. Clients provide the static map,
camera specs, initial frames, prompt, rig trajectory, and dynamic actor state
through the gRPC API. Use `--record_dir <dir>` to save replayable session logs,
and add `--enable_profiling --profile_output <path>` when collecting timing
data. For distributed/context-parallel launches, increase `--nproc_per_node`;
the world size must be compatible with the selected pipeline config's camera
count.
