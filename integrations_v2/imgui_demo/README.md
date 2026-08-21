<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui Demos

Three small v2 UI examples:

- `imgui-demo` renders one button from the ImGui-thread at ID `1` while
  the model-generation-thread at ID `0` uses `DISABLE_PRESENTATION`.
- `imgui-frame-sharing` rotates red, green, and blue model frames every
  ten model iterations. Thread `0` uses `HIDE_PRESENTATION`, so ImGui can draw
  its latest frame without blitting that frame directly to the client backbuffer.
- `imgui-message` prompts for the W key. The model-generation-thread
  receives the key event and uses `IThread.invoke_async` to change UI-owned text
  from `W is not Pressed` to `W is Pressed`.

## Usage

Install the demo and launch its WebRTC client:

```bash
uv sync --package flashdreams-imgui-demo --inexact
uv run --no-sync flashdreams-run-v2 imgui-demo --mode webrtc
uv run --no-sync flashdreams-run-v2 imgui-frame-sharing --mode webrtc
uv run --no-sync flashdreams-run-v2 imgui-message --mode webrtc
```

All three applications run model generation at 30 FPS and UI presentation at
60 FPS by default. The shared command's `--fps` and `--ui-fps` override those
rates independently.

Open the printed URL. CUDA, Vulkan/CUDA interop, SlangPy, and `imgui-bundle` are
required by the live renderer. The package's tests only construct the lazily
initialized renderer and remain CPU-safe.
