<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V2 framework tests

CPU-only tests for the v2 protocols themselves:

- `test_client_window.py` drives the I/O protocols against the deterministic NULL
  model integration.
- `test_session_runner.py` covers worker lifecycle, input delivery, reset
  generations, compositing, and window thread affinity.
- `test_thread.py` covers worker frequency validation, event-buffer fan-out, and
  the UI/ImGUI result wrapper.
- `test_mp4_client_window.py` covers lossless file-oriented session output and
  the optional benchmark metrics file written beside an MP4.
- `test_mp4_output_sink.py` and `test_metrics_output_sink.py` cover the two
  output sinks independently, including the runtime-stats schema consumed by
  the benchmark harness.
- `test_cli.py` covers `flashdreams-run-v2`, including `--stats-path` benchmark
  output and application discovery.
- `test_t2v_application.py`, `test_t2v_session.py`, and
  `test_t2v_model_check.py` cover finite text-to-video sessions and stand-ins.

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
