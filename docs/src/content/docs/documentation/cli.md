---
title: 'CLI'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

## Command Shape To Run A Demo/Model

The command shape is described in our [Running a model](../models/index.md#running-a-model) section:


```bash
uv run flashdreams-run-v2 <Demo Preset> <System Arguments> -- <Demo Arguments>
```

To request available arguments, specify `--help`:
```bash
uv run flashdreams-run-v2 --help
```

## System Arguments

### General

| Argument | Description |
| --- | --- |
| `--timeout SECONDS` | Stop after a finite number of seconds greater than zero. |
| `--mode {mp4,webrtc,native-window}` | Select file, browser, or local-window presentation. Default: `mp4`. |
| `--stats-path PATH` | Write JSON model-step measurements and enable synchronized per-stage pipeline profiling. |

### MP4 Specific

| Argument | Description |
| --- | --- |
| `--output-path PATH` | MP4 destination. Required in MP4 mode. |

### WebRTC Specific

| Argument | Description |
| --- | --- |
| `--host HOST` | Interface to serve on. Default: `127.0.0.1`; use `0.0.0.0` for remote clients. |
| `--port PORT` | Port to serve on. Default: `0`, which selects an available port. |

### Native Window Specific

| Argument | Description |
| --- | --- |
| `--window-title TITLE` | Native window title. Default: `FlashDreams`. |

### Default Demo Session Overrides

These options override the default demo session values requested by a particular demo implementation. Omit them to use the application's default values.

| Argument | Description |
| --- | --- |
| `--pixel-width N` | Override generated-frame width. |
| `--pixel-height N` | Override generated-frame height. |
| `--fps N` | Override the generated-frame playback rate. |
| `--layout {tchw,btchw,bcthw,bvtchw}` | Override generated tensor layout. |
| `--backpressure-mode {block,drop_oldest}` | Choose how the model thread handles a full presentation queue. |
| `--presentation-mode {on_demand,continuous}` | Choose when the presentation loop renders. |

#### Backpressure Mode

- `block` waits for queue capacity so generated chunks are retained.

- `drop_oldest` discards the oldest queued chunk to favor recent output.

#### Presentation Mode

- `on_demand` presents each selected model frame when it arrives.

- `continuous` keeps the UI responsive between model frames by reusing the newest frame.

### Examples

Stream to a browser:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode webrtc --host 0.0.0.0 --port 8089
```

Write every generated frame in order to an MP4:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode mp4 --output-path output.mp4 \
  --backpressure-mode block --presentation-mode on_demand
```

Open a local native window:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode native-window --window-title FlashDreams
```

## See also

- [Quickstart](../quickstart/index.md)
- [Configuration system](../developer_guides/config_system.md)
- [Runner slugs](../developer_guides/runner_slugs.md)
- [Infrastructure API](../api/infra.md)
