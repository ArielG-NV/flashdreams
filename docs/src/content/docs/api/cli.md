---
title: 'CLI'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams exposes two commands. `flashdreams-run` launches legacy runner
presets built on `flashdreams.infra` and applications registered under
`flashdreams.applications` using `flashdreams.runtime.demo`.
`flashdreams-run-v2` launches applications implementing the separate
`flashdreams.api_v2` protocol through `flashdreams.runtime_v2`.

## Inference runners

List the installed inference runner slugs and demo application slugs:

```bash

uv run flashdreams-run --help

```

Inspect one inference runner's full options:

```bash

uv run flashdreams-run RUNNER_SLUG --help

```

Run a single-GPU inference (`run` is the default launch mode):

```bash

uv run flashdreams-run RUNNER_SLUG

```

The common command shape is `flashdreams-run <runner> [mode]`. A runner only
advertises modes it implements; unsupported pairs fail before CUDA
initialization. Shared modes are `run`, `mp4`, `null`, `webrtc`, and
`local-window`.

Run a multi-GPU inference:

```bash

uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    RUNNER_SLUG

```

Resolve config only (no model instantiation):

```bash

uv run flashdreams-run --no-instantiate RUNNER_SLUG

```

## Demo API applications

Demo applications are registered under `flashdreams.applications` and use
the demo API for input/output modes, warmup, replay, benchmarking, and demo
validation. Their command shape is distinct from an inference runner:

```bash

uv run flashdreams-run DEMO_SLUG \
    --output mp4 --output-path output.mp4

```

Select `local-window`, `null`, `mp4`, or `webrtc` with `--output`.
For `mp4`, use `--output-path` and `--output-fps` as needed. For
`webrtc`, use `--host` and `--port`. Any other options are passed to the
demo application. The installed demo slugs appear under ``Installed application
demo slugs` in `flashdreams-run --help``.

## v2 applications

The v2 CLI lists its installed application slugs and runtime options:

```bash

uv run flashdreams-run-v2 --help

```

Launch the LingBot Cam2V application in a browser and pass `--example-data`
to the application after the `--` separator:

```bash

uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

The v2 presentation modes are `mp4` (the default), `webrtc`, and
`native-window`. Runtime options go before `--`; options after it belong to
the selected application. Use the following command to inspect those
application-specific options:

```bash

uv run flashdreams-run-v2 cam2v-lingbot -- --help

```

## Post-processing presets

Post-processing presets run on decoded RGB frames from a video runner. Select
one with `--postprocess.preset`:

```bash

uv run flashdreams-run RUNNER_SLUG \
    --postprocess.preset rtx-super-resolution

```

The `rtx-super-resolution` preset wraps NVIDIA VFX Python bindings for RTX
Video Super Resolution. Install the optional dependency with
`uv pip install 'flashdreams[rtx-postprocess]'` and run on a supported RTX GPU
before selecting this preset.

Native v2 applications receive their own arguments after `--`. Interactive
Drive exposes the equivalent setting with the hyphenated
`--postprocess-preset` option:

```bash

uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc -- \
    --postprocess-preset rtx-super-resolution

```

The preset starts enabled and can be toggled between generated chunks with the
Post-processing checkbox in the Interactive Drive HUD. Use
`flashdreams-run-v2 interactive-drive-omnidreams -- --help` to list the presets
registered in the current environment.

## See also

- [/quickstart/index](../quickstart/index.md)
- [/api/launch_manifests](launch_manifests.md)
- [/developer_guides/config_system](../developer_guides/config_system.md)
- [/developer_guides/runner_slugs](../developer_guides/runner_slugs.md)
- [/api/infra](infra.md)
