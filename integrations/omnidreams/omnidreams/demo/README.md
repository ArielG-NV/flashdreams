<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams applications

OmniDreams exposes two transport-neutral applications through the shared
FlashDreams host:

- `omnidreams` and `omnidreams-perf` replay precomputed HDMap conditioning.
- `interactive-drive` connects the reusable Interactive Drive app to the
  OmniDreams scene, simulation, and rendering runner.

The host owns local-window, null, MP4, and WebRTC I/O. Run commands from the
FlashDreams workspace root after installing the integration:

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>
uv sync --python 3.12 --package flashdreams-omnidreams --extra interactive-drive
```

## Precomputed replay

Run a finite replay without presentation:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams --output null \
  --device cuda:0 \
  --example-data \
  --example-data-uuid 239560dc-33d1-11ef-9720-00044bcbccac \
  --total-blocks 10
```

Write the same replay to MP4:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams --output mp4 \
  --output-path outputs/omnidreams-precomputed.mp4 \
  --output-fps 30 \
  --device cuda:0 \
  --example-data \
  --total-blocks 10
```

Use `omnidreams-perf` for the compile/CUDA-graph-oriented preset. Explicit
assets can be supplied with comma-separated `--hdmap-video-paths`,
`--first-frame-paths`, and `--camera-names` values.

## Interactive driving

The native local window is the default output:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  interactive-drive --auto-start --game-mode
```

Serve the same application over WebRTC:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  interactive-drive --output webrtc --host 0.0.0.0 --port 8089 \
  --auto-start --game-mode
```

Then open `http://localhost:8089/request_session`. Pass `--scene PATH` or
`--manifest PATH` to select a different scene/model configuration. The
application receives normalized throttle, brake, steer, stop, and reverse
values from whichever host input backend is selected.

Run `flashdreams-run omnidreams --help` or
`flashdreams-run interactive-drive --help` for application-specific options.
