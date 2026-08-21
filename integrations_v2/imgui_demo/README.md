<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui Demos

Three small v2 UI examples:

- `imgui-demo-webrtc` renders one button from auxiliary thread `1` while thread
  `0` uses `disablePresentation`.
- `imgui-frame-sharing-webrtc` rotates red, green, and blue model frames every
  ten model iterations. Thread `0` uses `hidePresentation`, so ImGui can draw
  its latest frame without blitting that frame directly to the client backbuffer.
- `imgui-message-webrtc` prompts for the W key. The model-generation thread
  receives the key event and uses `IThread.invoke_async` to change UI-owned text
  from `W is not Pressed` to `W is Pressed`.

## Usage

Install the demo and launch its WebRTC client:

```bash
uv sync --package flashdreams-imgui-demo --inexact
uv run imgui-demo-webrtc
uv run imgui-frame-sharing-webrtc
uv run imgui-message-webrtc
```

All three launchers run model generation at 30 FPS. `--fps` controls the UI and
presentation rate.

Open the printed URL. CUDA, Vulkan/CUDA interop, SlangPy, and `imgui-bundle` are
required by the live renderer. The package's tests only construct the lazily
initialized renderer and remain CPU-safe.
