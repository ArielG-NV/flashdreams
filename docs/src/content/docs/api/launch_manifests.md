---
title: 'Launch manifests'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Launch manifests configure registered runner presets and their launch
capabilities through this command shape:

```bash

uv run flashdreams-run <runner-slug> [mode] [--manifest PATH]

```

`run` is the default mode. Integrations may additionally expose `mp4`,
`null`, `webrtc`, or `local-window`. Inspect a resolved launch without
loading checkpoints or initializing CUDA with `--no-instantiate`.

For application commands, see [cli](cli.md); they do not use launch manifests.

## Schema

Launch manifests are strict, versioned YAML documents:

```yaml

schema_version: 1
runner: omnidreams
mode: webrtc

runner_overrides:
  device: cuda:0

scenario:
  scene_uuid: 0d404ff7-2b66-498c-b047-1ed8cded60d4
  scene_variant: default

output:
  host: 0.0.0.0
  port: 8089

```

`schema_version`, `runner`, and `mode` are required. The remaining
sections are optional mappings:

:::note

Quote the null-output mode as `mode: "null"` in YAML; an unquoted
`null` is YAML's null scalar rather than the FlashDreams mode name.

::: 

`runner_overrides`

: Recursive overrides for the registered runner configuration. The same
  runner fields remain available as explicit CLI flags.

`scenario`

: Inputs and controls such as prompts, example data, scenes, traces, and
  rollout length. The selected integration validates the accepted fields.

`output`

: Transport or artifact settings such as output path, frame rate, WebRTC
  bind address, warmup, and local-window presentation settings.

Relative paths in keys named `path` or `output`, or ending in `_path`,
`_paths`, or `_dir`, resolve relative to the manifest file, which makes
checked-in launch manifests reproducible from any working directory. Unknown
top-level or integration-specific fields fail before CUDA initialization.

## Precedence

Settings resolve in this order, from lowest to highest precedence:

```text

registered runner preset
  < manifest runner_overrides
  < manifest scenario/output
  < explicit CLI runner flags, --scenario.KEY/--output.KEY,
    and --host/--port

```

The runner and positional mode must agree with the manifest. For example, this
fails instead of silently launching a different preset:

```bash

uv run flashdreams-run <runner-slug> mp4 \
    --manifest path/to/webrtc-launch.yaml

```

## Application APIs

Demo API applications registered under `flashdreams.applications` are also
launched by `flashdreams-run`, but do not use launch manifests. Select their
output mode directly and pass application-specific arguments on the same
command line:

```bash

uv run flashdreams-run <application-slug> \
    --output mp4 --output-path outputs/demo.mp4 [application arguments]

```

Launch manifests configure `flashdreams-run` runners. V2 applications expose
their runtime and application arguments directly through `flashdreams-run-v2`
instead. For LingBot:

```bash

# MP4 replay
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode mp4 --output-path outputs/lingbot-replay.mp4 -- --example-data

# WebRTC
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

```

See [/models/lingbot_world](../models/lingbot_world.md) and [/models/omnidreams](../models/omnidreams.md) for their
application-specific arguments.
