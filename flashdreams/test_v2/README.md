<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V2 I/O tests

These CPU-only tests exercise the v2 I/O protocols with the deterministic NULL
model integration.

Run commands from the repository root.

## Run a focused test

```bash
uv run python -m pytest flashdreams/test_v2/test_client_window.py -v
```

To run every test in this directory:

```bash
uv run python -m pytest flashdreams/test_v2/ -v
```

The tests are marked `ci_cpu`; they do not require a GPU or model checkpoint.

## Windows cache permission errors

If `uv` cannot access its default cache, point it at a writable temporary
directory for the current PowerShell session:

```powershell
$env:UV_CACHE_DIR = Join-Path $env:TEMP "flashdreams-uv-cache"
uv run --package flashdreams-null-model python -m pytest flashdreams/test_v2/test_client_window.py -v
```
