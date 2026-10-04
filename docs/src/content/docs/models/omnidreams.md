---
title: 'NVIDIA OmniDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

[Blog page](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/)
[Tech report](https://arxiv.org/abs/2606.03159)
[Model page](https://huggingface.co/nvidia/omni-dreams-models/)
[Official code](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/omnidreams)

OmniDreams is an HDMap-conditioned streaming world model for driving
generation, with application configurations that balance visual fidelity and
runtime throughput.

See the [OmniDreams integration reference](../repository/integrations_v2/omnidreams/README.md) for registered variants and
application bindings.

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/teaser.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<p class="model-footnote">
  Teaser video source:
  <a href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/">OmniDreams project page</a>.
</p>

## Requirements

- **Minimum VRAM**: ~48 GB for the default Interactive Drive configuration.
- **PyTorch**: >= 2.11.
- **Python**: 3.10--3.12.

## Installation

```bash

# from the repo root
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact

```

Checkpoints and the default scene download from Hugging Face on first use.
Export `HF_TOKEN` when the selected repository requires authentication.

Generate an MP4 with the default Interactive Drive application:

```bash

uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams \
    --mode mp4 --output-path outputs/omnidreams.mp4 \
    --backpressure-mode block --presentation-mode on_demand \
    -- --no-ui --total-blocks 20

```

The command writes `outputs/omnidreams.mp4`. Arguments before `--` configure
the runtime and output mode; arguments after it configure Interactive Drive.
Run `flashdreams-run-v2 --help` and
`flashdreams-run-v2 interactive-drive-omnidreams -- --help` respectively to
inspect them.

## Application configurations

The OmniDreams integration registers these Interactive Drive application
slugs:

| Application slug | Description |
| --- | --- |
| `interactive-drive-omnidreams` | Default single-view, two-step HDMap-conditioned configuration. |
| `interactive-drive-omnidreams-optimized-gb300` | Benchmark-selected optimized attention policy for GB300. |
| `interactive-drive-omnidreams-optimized-rtx-pro-6000` | Benchmark-selected optimized attention policy for RTX PRO 6000. |
| `interactive-drive-omnidreams-perf` | Native DiT acceleration with the performance-tuned schedule. |
| `interactive-drive-omnidreams-fast-perf` | The performance configuration plus native FP8 LightVAE. |

These applications run on `flashdreams.runtime_v2`.

Some generated samples from the above commands:

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <!-- <div class="model-video-placeholder">Video placeholder</div> -->
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      example_data_uuid: "239560dc-33d1-11ef-9720-00044bcbccac"
    </div>
  </div>
  <div class="model-video-card">
    <!-- <div class="model-video-placeholder">Video placeholder</div> -->
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-24b84744-4156-11ef-b27d-00044bf655de-pip.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="model-video-overlay">
      example_data_uuid: "24b84744-4156-11ef-b27d-00044bf655de"
    </div>
  </div>
</div>

## Launch the interactive demo

OmniDreams exposes `webrtc`, `native-window`, and `mp4` through
`flashdreams-run-v2`. WebRTC only requires a CUDA-capable GPU;
`native-window` additionally requires a display and Vulkan.

The demo requires access to [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams)
and an `HF_TOKEN` with read access to
[nvidia/omni-dreams-scenes](https://huggingface.co/datasets/nvidia/omni-dreams-scenes)
(scene USDZs) and
[nvidia/omni-dreams-models](https://huggingface.co/nvidia/omni-dreams-models)
(checkpoints).

First-time setup:

```bash

git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
export HF_TOKEN=<your-hf-token>
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact

```

Optionally, pre-download scenes and checkpoints so the first launch
isn't blocked on network I/O:

```bash

uv run --no-sync omnidreams-prepare

```

Run the WebRTC demo:

```bash

uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams --mode webrtc \
    --host 0.0.0.0 --port 8089

```

Then open `http://<server-ip>:8089/` in any browser on the same network.

:::note

**The first launch is slow.** The first time you start the demo, the world
model spends several minutes in a one-time optimization pass -- checkpoint
loading, `torch.compile` / CUDA-graph capture, and Triton autotuning --
before the view becomes interactive. The on-screen indicator shows
`Loading world model...` during warmup and then `Optimizing world
model...` while the first generated chunk is autotuned; this phase is
longest on the perf configuration. Subsequent launches can be faster because
compiled kernels and autotuning results are cached; CUDA graphs are captured
again for each process.

::: 

On a GPU with a graphics stack, launch the Vulkan window:

```bash

uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams-perf --mode native-window

```

The native window's HUD adds a variant selector next to the scene picker, so a
scene can be switched among its published conditions (typically default,
rain, and snow).

:::note

The local window requires a display server and the system OpenGL /
Vulkan client libraries. On Debian/Ubuntu:

```bash
sudo apt install -y libx11-6 libxcb1 libgl1 libglx-mesa0 libvulkan1
```

A `Failed to initialize GLFW` error indicates the display or one of these
libraries are missing.

::: 

### Steering wheel and game controller

A steering wheel or game controller can be used to control native-window mode.
Any device that Ubuntu detects as a standard game controller
or joystick is viable. We provide a configuration tool to calibrate these:

```bash

uv run --package flashdreams-omnidreams interactive-drive-configuration

```

The demo auto-loads your default profile on subsequent launches. When you
have more than one profile, the configuration tool's start screen lists them
with **Make default** (plus Edit and Delete) buttons -- re-run the tool to
choose which profile `native-window` loads by default, tweak a profile
(steering sensitivity, deadzone, buttons, force feedback), or remove one.

**Multiple devices.** A profile can bind controls across several devices --
for example a wheel base plus a separately-connected or different-brand pedal
set. Ctrl+click to select more than one device on the configuration tool's
device page; each control binds to whichever selected device it moves on.

**Force feedback.** The method is auto-detected per wheel: a driver-managed
autocenter spring (Thrustmaster, Logitech) or a self-rendered constant force
(Fanatec, which has no autocenter). FFB needs the vendor's Linux driver and
write access to `/dev/input/*` (add your user to the `input` group):

| Vendor | Driver |
| --- | --- |
| Thrustmaster | Out-of-tree `hid-tmff2 &lt;https://github.com/Kimplul/hid-tmff2&gt;`__ plus a wheel-mode init (`hid-tminit`, or `tmdrv` for TX / TS-XW), for modern wheels (T300RS, T248, TX, T-GT II, TS-PC, TS-XW, …). |
| Fanatec | `hid-fanatecff &lt;https://github.com/gotzl/hid-fanatecff&gt;`__ with the base in PC mode (CSL DD, ClubSport, Podium, DD Pro). |
| Logitech | In-kernel `hid-lg4ff` or `new-lg4ff &lt;https://github.com/berarma/new-lg4ff&gt;`__ (G29, G27, G923 PS); the G920 and Xbox/PC G923 use the HID++ driver (kernel 6.3+). |

### Native acceleration (perf configuration)

The registered `interactive-drive-omnidreams-perf` configuration runs the DiT and
LightVAE through the OmniDreams single-view CUDA extension
(`native_dit_acceleration: required`), which is faster than the default
PyTorch path. The extension builds against pinned checkouts of CUTLASS,
SageAttention, SpargeAttn, and cudnn-frontend that are not vendored in the
repo. On first use, the native extension downloads them at their pinned commits
into `artifacts/omnidreams/thirdparty/` and then compiles (one-time, a few
minutes). It requires a source checkout (the `omnidreams_singleview` sources
ship only in the git tree, not the wheel), `git`, network access, and a CUDA
toolchain (`nvcc`) matching your PyTorch build. Then launch the perf
application:

```bash

uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams-perf --mode native-window

```

`native_dit_acceleration="required"` makes the perf config fail loudly if the
extension can't build or load, rather than silently falling back to PyTorch.

## WebRTC server

For browser deployments, the `webrtc` launch mode serves an HTML5 client on
top of the same OmniDreams pipeline.

```bash

# from the repo root
uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams --mode webrtc \
    --host 0.0.0.0 --port 8089

```

Sample scene UUIDs for the interactive server are available in the
[nvidia/omni-dreams-scenes Hugging Face dataset](https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes).
Scenes can publish default, rain, and snow variants as sibling archives. Pass
`-- --variant rain` or `-- --variant snow` to select one.

The server may take a few minutes to warm up. At startup, it prints the browser
URL, such as `Open http://<server-ip>:8089/ in a browser.`
Here, `<server-ip>` is the server IP address you are connecting to
(can use `localhost` when running locally).

:::note

On a remote or cloud GPU instance (e.g. [Brev](https://www.brev.dev/)),
the server port is usually not reachable at the host IP directly.
Forward it to your local machine first, then open
`http://localhost:8089/`:

```bash
# Brev
brev port-forward <instance> -p 8089:8089
# or plain SSH
ssh -L 8089:localhost:8089 <user>@<host>
```

::: 

Once successfully connected, the browser-based UI looks like this:

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-webrtc-recording-0529.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

:::note

If the page loads but the video never appears, the
browser is likely obfuscating local IPs in WebRTC ICE candidates
(replacing them with mDNS `.local` hostnames), which prevents the
peer connection from completing. Disable the setting and reload:

- **Chrome / Edge:** `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns` → **Disabled**, then restart the browser.
- **Brave:** `brave://settings/privacy/security` → *WebRTC IP handling policy* → **Default public and private interfaces**.
- **Firefox:** `about:config` → `media.peerconnection.ice.obfuscate_host_addresses` → **false**.

::: 

## Performance table

Single-view latency on NVIDIA GB300 at `704 x 1280` resolution.

| Stage | 1x GPU | 2x GPU | 4x GPU | 8x GPU |
| --- | --- | --- | --- | --- |
| HDMap Encoder | 28 ms | 26 ms | 26 ms | 26 ms |
| Diffusion DiT | 84 ms | 71 ms | 49 ms | 47 ms |
| VAE Decoder | 6 ms | 5 ms | 5 ms | 5 ms |
| KV-cache Update | 42 ms | 34 ms | 23 ms | 22 ms |
| **Total** | **118 ms** | **102 ms** | **80 ms** | **78 ms** |
| **Effective FPS** | **68** | **78** | **100** | **103** |

<p class="model-footnote">
   KV-cache Update is off the hot path and excluded from Total.
</p>

## Further reading

- [/developer_guides/latency_tuning](../developer_guides/latency_tuning.md) covers the supported
  `interactive-drive` latency knobs: model and backend choice, resolution,
  chunk-size constraints, FP8 and native acceleration, transport, and the
  validated GB300 reference.

## Citation

If you use OmniDreams, please cite the original work:

```bibtex

@misc{nvidia2026omnidreams,
  title         = {{NVIDIA} {OmniDreams}: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation},
  author        = {Basant, Aarti and Kar, Amlan and Paschalidou, Despoina and Wei, Fangyin and Ferroni, Francesco and Garcia Cobo, Guillermo and Turki, Haithem and Ling, Huan and Seo, Jaewoo and Lucas, James and Wu, Jay Zhangjie and Wang, Jialiang and Lorraine, Jonathan and Gao, Jun and He, Kai and Tothova, Katarina and Xie, Kevin and Tyszkiewicz, Micha{\l} and Wu, Qi and de Lutio, Riccardo and Li, Ruilong and Fidler, Sanja and Kim, Seung Wook and Shen, Tianchang and Cao, Tianshi and Pfaff, Tobias and Lew, William and Wu, Xindi and Ren, Xuanchi and Lu, Yifan and Zhang, Yuxuan and Gojcic, Zan and Wang, Zian},
  year          = {2026},
  eprint        = {2606.03159},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.03159},
  url           = {https://arxiv.org/abs/2606.03159},
}
```
