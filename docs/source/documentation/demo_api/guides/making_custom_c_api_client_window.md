---
title: 'Making a custom C API client window'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Quick start

Use `uv` to sync the `flashdreams-omnidreams` workspace package and its dependencies
(including Ninja) into the repository's `.venv`, preserving other installed
packages. Building the example requires `uv`, CMake 3.20+, a C compiler,
Bash, CPython 3.10+ development headers and embedding library, the Vulkan SDK
with SDL3 (`VULKAN_SDK` set), and the CUDA Toolkit. Running it requires
CUDA-capable PyTorch, an NVIDIA GPU with a compatible driver, and a desktop display.

From the repository root:

```bash
uv sync --locked --package flashdreams-omnidreams --inexact
bash c-api/examples/compile.sh
bash c-api/examples/run.sh
```

Run `uv sync` when installing or updating dependencies, compile when rebuilding
the C example, and run to launch the compiled example. The compile script
reconfigures the existing build directory, preserves incremental build outputs,
and accepts additional CMake configuration arguments.

CMake selects `.venv/bin/python` on Linux or `.venv/Scripts/python.exe` on
Windows. The run script uses the same interpreter selected at build time and
adds its directory to `PATH` so native extension builds can find tools such as Ninja.

The default example runs `crazy-robotaxi-omnidreams` in a resizable SDL3 window
using the C client backend. It forwards keyboard, mouse, and focus events for
the menus and driving controls. The window stays open until you close it or
exit the demo; the example injects no input and sets no timeout.
Follow the [Crazy Robotaxi setup guide](../../../../../apps/crazy_robotaxi/README.md)
for model assets, GPU requirements, and `HF_TOKEN` access. First launch may
download assets and compile kernels.

To select another preset, pass explicit launch arguments:

```bash
bash c-api/examples/run.sh crazy-robotaxi-omnidreams-rtx-5090 --mode c
```

## What it does and how it works

`flashdreams_capi` embeds CPython so a C or C++ host can run an existing v2 app
through `fd_run`. It reuses app discovery, argument parsing, the session runner,
and cleanup. The host supplies window callbacks for input and presentation;
FlashDreams keeps its model thread, while callbacks run serially on the calling/UI
thread. The SDL3 example copies CUDA frames to a software window surface.

To build your own host, include `c-api/include/flashdreams.h`, link
`flashdreams_capi`, and:

1. Initialize `fd_client_window` with `struct_size = sizeof(window)`,
   `abi_version = FD_ABI_VERSION`, and the required `open`,
   `get_user_input_events`, `write`, and `close` callbacks. Cursor and resize
   callbacks are optional.
2. Call `fd_run` with the Python environment's absolute executable path, launch
   arguments without a program name, the callback table, and `--mode c`.
   Passing `NULL` for the table uses regular window modes instead.
3. Keep the table, context, and callback code alive until `fd_run` returns.
   It blocks and returns `0` for success/help or `1` for failure, with an optional
   error buffer. Call it serially; callbacks must not throw or reenter it.

`get_user_input_events` returns `1` for one consumed event, `0` when drained,
or `-1` on failure. Events cover close, reset, keyboard, mouse, and focus.
`open` and `write` return zero on success; `close` must tolerate repeated calls
and partially failed initialization. See the header for event fields and details.

`write` receives a borrowed `fd_cuda_frame`: packed CUDA uint8 RGB24 pixels in
`[frames][height][width][3]` order, dimensions, device, and stream. GPU output
stays on its device; CPU output is uploaded. There is no CPU delivery fallback.

Enqueue reads or copies on the supplied stream **before returning from `write`**;
the bridge keeps the allocation alive until that work completes. A different
consumer stream must wait for production and join back onto the supplied stream
before return. Do not dereference the pointer on the CPU, free it, destroy the
stream, or retain the descriptor. Copy into host-owned memory if needed later.

Python stays initialized between runs. If the host already initialized Python,
`fd_run` uses that interpreter. Do not finalize Python or unload the library
during a run.

### Tests

Standalone C API checks live in `c-api/tests/`; Python bridge tests live in
`flashdreams/test_v2/`. The example's input check uses SDL's dummy video driver
and needs no model or GPU work:

```bash
cmake --build c-api/build --config Release --target test_example_input
./c-api/build/test_example_input
```

With a multi-configuration generator on Windows, run
`c-api/build/Release/test_example_input.exe` instead.

CPU tests check the bridge contract; CUDA tests cover frame delivery, stream
ordering, buffer lifetime, and failure cleanup (skipping without CUDA):

```bash
uv run --group test pytest flashdreams/test_v2/test_c_api.py -m ci_cpu
uv run --group test pytest flashdreams/test_v2/test_c_api_cuda.py -m ci_gpu
```

On Windows, `.venv/Scripts/python.exe c-api/tests/test_window.py` checks
visible output, resizing, keyboard input, and closing; it requires Pillow.
It is marked `manual` in pytest so CPU CI does not open a desktop window.
