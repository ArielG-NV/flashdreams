<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui Demos

Two small v2 UI examples:

- `imgui-demo-webrtc` renders one button from auxiliary thread `1` while thread
  `0` is disabled.
- `imgui-frame-sharing-webrtc` shows thread `0`'s last-presented frame inside an
  ImGui window. The UI worker reads the frame and sends a color-change message
  back to the model-generation worker through the `IThread` API.

## Usage

Install the demo and launch its WebRTC client:

```bash
uv sync --package flashdreams-imgui-demo --inexact
uv run imgui-demo-webrtc
```

Open the printed URL. CUDA, Vulkan/CUDA interop, SlangPy, and `imgui-bundle` are
required by the live renderer. The package's tests only construct the lazily
initialized renderer and remain CPU-safe.
