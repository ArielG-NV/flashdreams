---
title: 'Cosmos3 Nano Sim Bimanual'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Run [HandDreams](../demos/handdreams.md) with a browser or native viewer, using
virtual controllers or a USB-connected Quest.

## Requirements

- Ubuntu x86_64/aarch64, glibc 2.35+, Python 3.12, Git, `uv`, and OpenSSL.
- An NVIDIA GPU with enough memory for the model, CUDA, Vulkan, and **NVENC**.
  RTX PRO 6000 has been tested. These commands run CloudXR on the server;
  a remote-VR laptop also needs NVENC and Vulkan.
- Chrome/Chromium with working WebGL and graphics acceleration for browser use.
- Your own Hugging Face access to [the model](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)
  and [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta).
  Complete their access requirements and review their terms.
- Internet for first-time setup. Model shards and the merged cache use about
  **61 GB**, plus space for dependencies, decoder weights, and scenes.

## Quick Start

### HandDreams

Run from the repository root on the **server**. Omit the token export if already
logged in to Hugging Face. Setup displays the CloudXR license; type `yes` if you accept.

```bash
uv sync --python 3.12 --package flashdreams-cosmos3-nano-sim-bimanual --inexact
export HF_TOKEN="<YOUR_HF_TOKEN>"
uv run --no-sync flashdreams-controller-input setup isaac-teleop
```

Choose one launch command below. Include EULA flags only if you accept the terms.

### WebRTC (Local or Remote)

Use `127.0.0.1` for a browser on the server, or set `SERVER_IP` to the server's
reachable IPv4 address for a remote browser:

```bash
SERVER_IP=127.0.0.1
uv run --no-sync flashdreams-run-v2 handdreams \
  --mode webrtc --host "$SERVER_IP" --port 8081 \
  --controller-host "$SERVER_IP" \
  --isaac-teleop-accept-eula
```

Open `http://<SERVER_IP>:8081/`, then
[connect virtual controllers](../demos/handdreams.md#connect-virtual-controllers).
Use a trusted network: the viewer (`8081`), Teleop HTTPS (`48322`), and their
WebRTC media traffic must be reachable. `0.0.0.0` is not a browser destination.

### Native Window

Run from the **server's graphical desktop**:

```bash
uv run --no-sync flashdreams-run-v2 handdreams \
  --mode native-window \
  --isaac-teleop-accept-eula
```

Click **Isaac Teleop Web Client: Open Link** in the window and follow the
[connection steps](../demos/handdreams.md#connect-virtual-controllers), using
`127.0.0.1`. Native mode changes the viewer, not controller input; there is no
viewer page on port `8081`.

## Physical Quest Controllers

On the Ubuntu machine connected to the Quest by USB:

```bash
sudo apt install adb coturn
adb devices
```

Enable Quest Developer Mode and authorize USB debugging; the device must show
as `device`, not `unauthorized`. Keep Quest Wi-Fi connected with an IPv4 address.
An offline access point is sufficient; the controller stream uses USB.
Generated video stays in the demo viewer. No custom APK or coturn service is needed.

The commands below clear the legacy `QUEST_META_PIN` setting and permit
certificate errors only for the managed tab's local CloudXR origin, not the
remote relay. Startup restarts Quest Browser, closing its other sessions.

### Headset Connected to the Server

Complete server setup and connect the Quest **before** launch. A single authorized
headset is detected automatically; use `--quest-device <SERIAL>` to select one.

For native output:

```bash
env -u QUEST_META_PIN uv run --no-sync flashdreams-run-v2 handdreams \
  --mode native-window \
  --isaac-teleop-accept-eula \
  --quest-accept-server-certificate
```

For WebRTC output, set the server address and run this instead:

```bash
SERVER_IP=127.0.0.1
env -u QUEST_META_PIN uv run --no-sync flashdreams-run-v2 handdreams \
  --mode webrtc --host "$SERVER_IP" --port 8081 \
  --controller-host "$SERVER_IP" \
  --isaac-teleop-accept-eula \
  --quest-accept-server-certificate
```

Use the server's LAN address for a remote browser. Open `http://<SERVER_IP>:8081/`
for WebRTC, then follow [physical-controller setup](../demos/handdreams.md#physical-controller-setup).
The headset browser connects automatically; do not also connect an IWER client.

### Headset Connected to a Laptop

On the **server**, start [WebRTC](#webrtc-local-or-remote) with its reachable
LAN address. Leave it running.

On the **Ubuntu laptop**, use a matching checkout, meet the Quest requirements
above, and install Isaac Teleop once from the repository root:

```bash
uv run --no-project --python 3.12 python \
  integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py setup-isaac-teleop
```

The laptop needs `ssh`/`scp`, a verified server host key, and key/agent login.
Check that this succeeds without a password prompt:

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

It prepares controller dependencies and copies credentials and web assets over
SSH; model weights stay on the server. The laptop also needs direct access to
server TCP `8445`—SSH does not tunnel it.

Open `http://<SERVER_IP>:8081/` and follow
[physical-controller setup](../demos/handdreams.md#physical-controller-setup).
For native output on the server, change its mode to `native-window` but retain
`--controller-host "<SERVER_IP>"`; the laptop command is unchanged.

## Scene and Model Defaults

The demo downloads **Fold the T-shirt**, **Fold the shorts**, and **Open the tissue
box** from AgiBotWorld-Beta using your credentials. Scene data is not bundled.
Defaults are the GA checkpoint, four denoising steps, and TAEHV2 Super decoder,
producing 640 × 352 video. No checkpoint override or S3 credentials are needed.

For your own catalog, append `-- --sample-dir /absolute/path/to/scenes` to the
server command. Up to five subdirectories are supported, each with `first_frame.jpg`
and `scene.json`. For one scene, use `--scene-path /absolute/path/to/scene` after
`--` instead. These paths are on the server.

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
| Virtual controllers slide or jump | Use the [toolbar reset workaround](../demos/handdreams.md#reset-virtual-controllers). |
| “Connection Lost” or XR CONTROL flickers | Check the Teleop tab and capture the corresponding server error; the viewer message alone does not identify the cause. |
| Video degrades during a long rollout | Press Escape and select the scene again to start a fresh rollout. |

If it persists, share the commit, command, GPU/driver, browser version, and first
relevant error. Remove credentials from logs.

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
