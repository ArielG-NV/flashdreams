---
title: 'HandDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

HandDreams generates robot-manipulation video from virtual or physical
controllers. It does not control a physical robot.

## Quick Start

<div class="fd-cta-row" markdown="span">
  [Cosmos3 Nano Sim Bimanual](../models/cosmos3_nano_sim_bimanual.md#handdreams){ .fd-button }
</div>

## Connect Virtual Controllers

No headset is needed. After starting the server:

1. Open `http://<SERVER_IP>:8081/`, or use the native viewer. For local use,
   `<SERVER_IP>` is `127.0.0.1`.
2. Click **Isaac Teleop Web Client: Open Link** in the viewer.
3. Accept the development certificate after verifying the origin is your
   server's `https://<SERVER_IP>:48322/`. If the client shows an accept-certificate
   link, follow it and return to the Teleop tab.
4. Allow local-network access if requested. Keep the prefilled settings and
   click **CONNECT**.
5. Confirm both controller panels appear and the viewer reports both tracked.
   Release both triggers and keep the Teleop tab open alongside the viewer.

If you changed client settings, restore reference space `local`, zero pose
offsets and prediction, and no smoothing.

## Start a Scene

Click a thumbnail, or use the viewer's arrow keys followed by **Enter**.

### Virtual-Controller Setup

Hold either **Trig** until the progress bar fills **in the demo viewer**, then
release both triggers. No alignment is needed. Use the panels' position and
rotation controls to move the hands.

### Physical-Controller Setup

[Connect the Quest](../models/cosmos3_nano_sim_bimanual.md#physical-quest-controllers),
press its **power button** when prompted, and approve any headset permissions.
Watch the demo viewer for calibration:

1. Hold the controllers side by side and squeeze both **index triggers** until
   the progress bar fills and gray guides appear.
2. Release both triggers. Move and rotate the controllers to match the guides,
   keeping both claws open until alignment completes.

## During the Demo

- Increase virtual **Trig** or squeeze a physical index trigger to close its
  gripper; release to open. Move the controllers to move the hands.
- **Escape** returns to scene selection.
- **XR CONTROL** shows controller input, not the generated robot's pose.
- Keyboard **B** in the viewer retries a server-attached Quest, including the
  power-button wake. For a laptop Quest, restart `remote-vr` instead.
- **Ctrl+C** stops the server. Stop `remote-vr` on the laptop too; cleanup
  requests sleep on the managed headset.

Disconnect the current input client before connecting another; switching
sources repeats calibration.

## Demo Arguments

Place these after the runtime command's `--` separator.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--position-gain` | `1.2` | Scale controller translation. |
| `--control-fps` | Scene FPS | Input sampling and model-conditioning cadence, not display refresh. |
| `--auto-start` | Off | Skip selection and guided setup; requires both controllers tracked with open grippers. |

## Troubleshooting

### Reset Virtual Controllers

If a controller keeps sliding after mouse release, in the **Isaac Teleop toolbar**:

1. Click the **game-controller icon** to switch to hands.
2. Click the **hand icon** to switch back to controllers.
3. Confirm both controllers are tracked. If positions changed, press **Escape**
   in the viewer and select the scene again.

This is not the widget's **Reset** button. If it fails, reopen the client and
reconnect. Small movements with individual axis arrows were more reliable than
plane dragging in testing.

For other issues, see [troubleshooting](../models/cosmos3_nano_sim_bimanual.md#troubleshooting).
