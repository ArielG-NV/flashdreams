---
title: 'Waypoint Action2V'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-waypoint-apps-action2v-readme--waypoint-action2v"></a>

<a id="integrationsv2-waypoint-apps-action2v-readme--integration-links"></a>

## Integration links

- **Applications:** [Shared Action2V](../../../../apps/action2v/README.md)
- **Configuration:** [`config.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/waypoint/config.py)

This adapter binds the Waypoint model in `flashdreams-waypoint` to the shared
Action2V application.
Model modules are loaded once per application; the image-established
transformer/decoder cache, RNG stream, and live controls are isolated per
session.

The application uses the shared [apps/action2v](../../../../apps/action2v/README.md)
shell.

<a id="integrationsv2-waypoint-apps-action2v-readme--command-line-options"></a>

## Command-line options

Arguments after `--` belong to the Action2V application; v2 runtime options go
before it. Waypoint uses the shared Action2V options and adds no model-specific
flags.

| Option | Default | Description |
| --- | --- | --- |
| `--image-path PATH` | unset | Use an RGB or RGBA image to establish the world. |
| `--example-data`, `--no-example-data` | off | Enable or disable the pinned public example image. |
| `--device DEVICE` | `cuda` | Select the model device. |
| `--total-blocks N` | `10000` | Stop after this many generated action blocks. |
| `--ui`, `--no-ui` | on | Enable or disable interactive pointer capture. |
| `--seed N` | `42` | Override the model RNG seed for reproducible generation. |
| `--mouse-sensitivity SCALE` | `1.0` | Scale normalized pointer motion before mapping it to Waypoint pixel deltas. |
| `--reset-key CHAR` | `T` | Reset the current session when this ASCII letter is pressed. |
| `-h`, `--help` | — | Print these options without loading checkpoints. |

Run `flashdreams-run-v2 --help` for runtime options such as `--mode`, the MP4
`--output-path`, the profiling `--stats-path`, and the WebRTC `--host` and
`--port`.

<a id="integrationsv2-waypoint-apps-action2v-readme--run-waypoint"></a>

## Run Waypoint

For live keyboard and mouse input, use the browser window and the example image:

```bash

uv run flashdreams-run-v2 action2v-waypoint-1-5-1b --mode webrtc \
    -- --example-data --seed 464

```

Arguments after `--` belong to the Action2V application. Run
`flashdreams-run-v2 action2v-waypoint-1-5-1b -- --help` for the complete
list. The first `--example-data` run downloads the public image; the first model
run downloads the Waypoint 1.5 1B and TAEHV checkpoints.

The application declares four-frame `TCHW` results on Waypoint's native
1024x512 canvas at 60 FPS playback. Input images are resized once to that native
canvas; generated frames are presented without another spatial resample.
