---
title: 'Cosmos3 Nano Sim Bimanual'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual">Checkpoint and model card</a>
  <a class="fd-button" href="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta">Scene dataset</a>
</div>

Cosmos3 Nano Sim Bimanual generates robot-manipulation video from a first-frame image,
prompt, and controller actions. Try it with [HandDreams](../demos/handdreams.md).

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/demo_clips/handdreams.avif" alt="HandDreams robot hands folding a T-shirt." />
</div>
<figcaption class="tiny-figcaption">
  HandDreams demo with a custom scene catalog.
</figcaption>

## Quick Start

### HandDreams

Check the [requirements](#requirements), then run from the repository root on
the **server**. Setup displays the CloudXR license; type `yes` and use EULA
flags below only if you accept the terms.

<a id="webrtc-local-or-remote"></a>

```bash
uv sync --python 3.12 --package flashdreams-cosmos3-nano-sim-bimanual --inexact
export HF_TOKEN="<YOUR_HF_TOKEN>"  # Omit if already logged in to Hugging Face
uv run --no-sync flashdreams-controller-input setup isaac-teleop

SERVER_IP=127.0.0.1  # Use the server's reachable LAN IPv4 for a remote browser
uv run --no-sync flashdreams-run-v2 handdreams \
  --mode webrtc --host "$SERVER_IP" --port 8081 \
  --controller-host "$SERVER_IP" \
  --isaac-teleop-accept-eula
```

Open `http://<SERVER_IP>:8081/`, then
[connect virtual controllers](#connect-virtual-controllers).

<div class="fd-cta-row">
  <a class="fd-button" href="../demos/handdreams.md#controls">Controls</a>
  <a class="fd-button" href="#handdreams-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/handdreams.md#demo-arguments">Demo Arguments</a>
</div>

Other setups: [native window](#native-window) · [physical Quest controllers](#physical-quest-controllers).

<hr>

## Demo Presets

<a id="handdreams-presets"></a>

### HandDreams

| Preset | Description |
| --- | --- |
| `handdreams` | GA checkpoint with four denoising steps and the TAEHV2 Super decoder, producing 640 × 352 video. |

No checkpoint override or S3 credentials are needed.

<hr>

## Developer Details

### Requirements

- Ubuntu x86_64/aarch64, glibc 2.35+, Python 3.12, Git, `uv`, and OpenSSL.
- An NVIDIA GPU with enough model memory, CUDA, Vulkan, and **NVENC**.
  RTX PRO 6000 has been tested. CloudXR runs on the server; a remote-VR laptop
  also needs NVENC and Vulkan.
- Chrome/Chromium with working WebGL and graphics acceleration for browser use.
- Your own Hugging Face access to [the model](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)
  and [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta).
  Complete their access requirements and review their terms.
- Internet for first-time setup. Model shards and the merged cache use about
  **61 GB**, plus dependencies, decoder weights, and scenes.

### Connect Virtual Controllers

No headset is needed. After starting the server:

1. Open `http://<SERVER_IP>:8081/`, or use the native window. For local use,
   `<SERVER_IP>` is `127.0.0.1`.
2. Click **Isaac Teleop Web Client: Open Link** in the viewer.
3. Verify the certificate origin is your server's `https://<SERVER_IP>:48322/`,
   then accept its development certificate. Follow any accept-certificate link
   and return to the Teleop tab.
4. Allow local-network access if requested. Keep the prefilled settings and
   click **CONNECT**.
5. Confirm both controller panels appear and the viewer reports both tracked.
   Release both triggers and keep the Teleop tab open alongside the viewer.

Continue with [scene setup and controls](../demos/handdreams.md#controls).
If client settings changed, restore reference space `local`, zero pose offsets
and prediction, and no smoothing.

Use a trusted network with viewer TCP `8081`, Teleop HTTPS `48322`, and WebRTC
media traffic reachable. `0.0.0.0` is not a browser destination.
Disconnect the current input client before connecting another; switching
sources repeats calibration. **Ctrl+C** stops the server. Also stop `remote-vr`
on the laptop; cleanup requests sleep on the managed headset.

### Native Window

After the setup commands above, run from the **server's graphical desktop**:

```bash
uv run --no-sync flashdreams-run-v2 handdreams \
  --mode native-window \
  --isaac-teleop-accept-eula
```

Click **Isaac Teleop Web Client: Open Link** in the window and follow the
[connection steps](#connect-virtual-controllers) with `127.0.0.1`.
There is no viewer page on port `8081`.

### Physical Quest Controllers

On the Ubuntu machine connected to the Quest by USB:

```bash
sudo apt install adb coturn
adb devices
```

Enable Quest Developer Mode and authorize USB debugging; the device must show
as `device`. Keep Quest Wi-Fi connected with an IPv4 address; an offline access
point works. Controllers stream over USB; generated video stays in the demo
viewer. No custom APK or coturn service is needed.

These commands clear legacy `QUEST_META_PIN` and permit certificate errors
only for the managed tab's local CloudXR origin, not the remote relay.
Startup restarts Quest Browser, closing its other sessions.

#### Headset Connected to the Server

Complete [server setup](#handdreams) and connect the Quest **before** launch.
A single authorized headset is detected automatically; select one with
`--quest-device <SERIAL>` if needed.

For native output:

```bash
env -u QUEST_META_PIN uv run --no-sync flashdreams-run-v2 handdreams \
  --mode native-window \
  --isaac-teleop-accept-eula \
  --quest-accept-server-certificate
```

For WebRTC output:

```bash
SERVER_IP=127.0.0.1
env -u QUEST_META_PIN uv run --no-sync flashdreams-run-v2 handdreams \
  --mode webrtc --host "$SERVER_IP" --port 8081 \
  --controller-host "$SERVER_IP" \
  --isaac-teleop-accept-eula \
  --quest-accept-server-certificate
```

Use the server's LAN IPv4 for a remote browser and open `http://<SERVER_IP>:8081/`.
The headset browser connects automatically; do not also connect a virtual
client. Continue with [physical-controller setup](../demos/handdreams.md#physical-controller-setup).

#### Headset Connected to a Laptop

On the **server**, start [WebRTC](#handdreams) with its reachable LAN IPv4 and
leave it running.

On the **Ubuntu laptop**, use a matching checkout, meet the requirements above,
and install Isaac Teleop once from the repository root:

```bash
uv run --no-project --python 3.12 python \
  integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py setup-isaac-teleop
```

With `ssh`/`scp`, a verified server host key, and key/agent login, check that
this succeeds without a password prompt:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes "<USER>@<SERVER_IP>" true
```

Start the laptop client:

```bash
env -u QUEST_META_PIN uv run --no-project --python 3.12 python \
  integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py remote-vr \
  --server "<USER>@<SERVER_IP>" \
  --accept-eula \
  --accept-server-certificate
```

This prepares controller dependencies and copies credentials and web assets
over SSH; weights stay on the server. The laptop also needs direct access to
server TCP `8445`; SSH does not tunnel it.

Open `http://<SERVER_IP>:8081/` and follow
[physical-controller setup](../demos/handdreams.md#physical-controller-setup).
For native output on the server, change its mode to `native-window` but retain
`--controller-host "<SERVER_IP>"`; the laptop command is unchanged.

### Scene Data

The demo downloads **Fold the T-shirt**, **Fold the shorts**, and **Open the tissue
box** from AgiBotWorld-Beta with your credentials; scene data is not bundled.

For a custom catalog, append `-- --sample-dir /absolute/path/to/scenes` to the
server command: up to five subdirectories, each with `first_frame.jpg` and
`scene.json`. For one scene, use `--scene-path /absolute/path/to/scene` after
`--`. Paths refer to the server.

<hr>

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Startup looks stuck | First launch downloads assets and warms up scenes, sometimes with little output. Check the terminal for progress or an error; let active work finish. |
| Hugging Face returns 401/403 or a gated-repository error | Check access to both repositories and the server shell's `HF_TOKEN`. |
| Home quota exceeded | Check cache space; see [storage settings](#limited-home-storage). |
| `NV_ENC_ERR_UNSUPPORTED_DEVICE` | Use an NVENC-equipped host. A GB300-only host cannot run the CloudXR configuration documented here. |
| “Bridge failed” | Read the preceding error and check `nvidia-smi`. If a pending driver update or driver/library mismatch is confirmed, complete the update and reboot. |
| “Immersive Mode Not Supported” | Check `chrome://gpu` for disabled WebGL or GPU crashes; enable working graphics acceleration. |
| Viewer or Teleop cannot connect | Check the address, certificate, browser local-network permission, and firewall. HTTP/HTTPS access alone does not establish WebRTC media connectivity. |
| Remote-VR cannot obtain credentials | Start the server and run the strict SSH check above. The launcher does not create SSH keys or accept unknown hosts. |
| Headset never connects | Check USB authorization, Quest Wi-Fi, controller batteries, and headset prompts. See [retry controls](../demos/handdreams.md#during-the-demo). |
| Virtual controllers slide or jump | Use the [toolbar reset workaround](#reset-virtual-controllers). |
| “Connection Lost” or XR CONTROL flickers | Check the Teleop tab and capture the corresponding server error; the viewer message alone does not identify the cause. |
| Video degrades during a long rollout | Press Escape and select the scene again to start a fresh rollout. |

If it persists, share the commit, command, GPU/driver, browser version, and first
relevant error. Remove credentials from logs.

### Reset Virtual Controllers

If a controller keeps sliding after mouse release, in the **Isaac Teleop toolbar**:

1. Click the **game-controller icon** to switch to hands.
2. Click the **hand icon** to switch back to controllers.
3. Confirm both controllers are tracked. If positions changed, press **Escape**
   in the viewer and select the scene again.

This is not the widget's **Reset** button. If it fails, reopen the client and
reconnect. Small movements with individual axis arrows were more reliable than
plane dragging in testing.

### Limited Home Storage

Before installation, set `HF_HOME`, `FLASHDREAMS_CACHE_DIR`, and `UV_CACHE_DIR` to
larger writable directories. Put the checkout there too for its `.venv`.
These settings do not move existing files.

To install Isaac Teleop elsewhere:

```bash
uv run --no-sync flashdreams-controller-input setup isaac-teleop \
  --directory /absolute/path/on/large-volume/isaac-teleop
```

Add `--isaac-teleop-python /absolute/path/on/large-volume/isaac-teleop/bin/python`
**before and after `--`** on the server: CloudXR and calibration have separate
options. On the laptop, add it once to `remote-vr`.
