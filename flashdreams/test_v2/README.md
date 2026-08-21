<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V2 framework tests

CPU-only tests for the v2 protocols themselves:

- `test_client_window.py` covers client-window I/O against the NULL model.
- `test_application_runner.py` covers application, session, and window lifecycle.
- `test_session_runner.py` covers user-visible-threads, events, presentation,
  and shutdown.
- `test_imgui_thread.py` covers ImGUIThread implementation, testing input routing and renderer.
- `test_mp4_client_window.py` covers file-window output and benchmark metrics.
- `test_mp4_output_sink.py` covers MP4 encoding and output validation.
- `test_client_window_factory.py` covers presentation-mode arguments and selection.
- `test_webrtc_client_window.py` covers browser input and video-frame delivery.
- `test_t2v_application.py` covers shared text-to-video application behavior.
- `test_t2v_session.py` covers text-to-video rollout, reset, and cleanup.
- `test_t2v_model_check.py` covers text-to-video integration failure reporting.
- `test_cli.py` covers v2 application discovery, arguments, and output modes.
- `test_metrics_output_sink.py` covers the benchmark runtime-metrics schema.

Every `test_*.py` file in this directory must have one entry in this list.

Application behaviour is tested by the application that owns it — see
`integrations_v2/red_screen/red_screen/tests/` and
`integrations_v2/color_fade/color_fade/tests/`.

Run commands from the repository root.

## Set up the test environment

```bash
uv sync --package flashdreams-color-fade --package flashdreams-red-screen --package flashdreams-null-model --group test --inexact
```

`test_client_window.py` imports the NULL model integration, and naming both
integrations leaves the environment ready for their tests too. `--inexact`
matters: without it, `uv` makes the environment exact for the packages it was
given and uninstalls the rest. `pytest` comes from the `test` group; do not use
`--extra dev`, which pulls `transformer-engine` and compiles CUDA extensions from
source.

## Run the tests

```bash
uv run --no-sync pytest flashdreams/test_v2 -m ci_cpu -v
```

A single test:

```bash
uv run --no-sync pytest flashdreams/test_v2/test_session_runner.py -v
```

`--no-sync` keeps the run from re-resolving the environment.

The tests are marked `ci_cpu`; they need no GPU and no model checkpoint.
