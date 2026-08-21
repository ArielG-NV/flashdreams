<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui Demo

Minimal v2 integration with no model output. Main-generation thread `0` returns
an explicitly disabled `StepResult`; auxiliary thread `1` draws a Dear ImGui
window and supplies the only visible layer.

The browser forwards normalized pointer movement, mouse buttons, wheel input,
keyboard edges, and focus changes. Printable key presses feed the active text
widget selected by a mouse click. The window includes a
checkbox, slider, text field, and button so each input path can be exercised.

## Usage

Install the demo and launch its WebRTC client:

```bash
uv sync --package flashdreams-imgui-demo --inexact
uv run imgui-demo-webrtc
```

Open the printed URL. CUDA, Vulkan/CUDA interop, SlangPy, and `imgui-bundle` are
required by the live renderer. The package's tests only construct the lazily
initialized renderer and remain CPU-safe.
