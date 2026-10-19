---
title: 'HandDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Control generated robot hands with virtual or physical controllers. No physical
robot is controlled.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/demo_clips/handdreams.avif" alt="HandDreams robot hands folding a T-shirt." />
</div>
<figcaption class="tiny-figcaption">
  HandDreams demo with a custom scene catalog.
</figcaption>

## Quick Start

<div class="fd-cta-row" markdown="span">
  [Cosmos3 Nano Sim Bimanual](../models/cosmos3_nano_sim_bimanual.md#handdreams){ .fd-button }
</div>

<hr>

## Controls

After [connecting controllers](../models/cosmos3_nano_sim_bimanual.md#connect-virtual-controllers),
click a scene thumbnail or use **arrow keys → Enter**.

### Virtual-Controller Setup

Hold either **Trig** until the demo viewer's progress bar fills, then release
both triggers. No alignment is needed.

### Physical-Controller Setup

After [Quest setup](../models/cosmos3_nano_sim_bimanual.md#physical-quest-controllers),
press the headset's power button when prompted and approve its permissions.
Hold the controllers side by side and squeeze both index triggers until gray
guides appear. Release, then match the guides with both claws open.

### During the Demo

| Input | Action |
| --- | --- |
| Panel position/rotation or physical controller movement | Move the hands. |
| **Trig** / index trigger | Squeeze to close; release to open. |
| **Escape** | Return to scene selection. |
| Keyboard **B** | Retry a server-attached Quest. For a laptop Quest, restart `remote-vr`. |

XR CONTROL shows controller input, not the generated pose.

<hr>

## Demo Arguments

Append after `--` in the launch command.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--position-gain` | `1.2` | Scale controller translation. |
| `--control-fps` | Scene FPS | Input sampling rate, not display refresh. |
| `--auto-start` | Off | Skip selection and guided setup; requires tracked controllers and open grippers. |

See [connection troubleshooting](../models/cosmos3_nano_sim_bimanual.md#troubleshooting)
or the [virtual-controller reset workaround](../models/cosmos3_nano_sim_bimanual.md#reset-virtual-controllers).
