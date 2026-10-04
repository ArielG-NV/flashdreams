---
title: 'V2 framework tests'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="flashdreams-testv2-readme--v2-framework-tests"></a>

Tests for `flashdreams.api_v2` and `flashdreams.runtime_v2`.

Most tests in this directory are CPU-only:

- `test_client_window.py` drives the I/O protocols against the deterministic NULL
  model integration.
- `test_session_runner.py` covers the `run_session` loop with fake session and
  window implementations, so it depends on no integration at all. It asserts the
  orderings and thread ownership the two-thread loop guarantees, not a particular
  interleaving.
- `test_application_runner.py` covers application initialization, session
  selection, cleanup, and metrics handling around `run_session`.
- `test_mp4_client_window.py` covers the window a run writing a file is driven
  against: no input to report, one run through the loop to show every step
  reaches the file, and the measurements it records beside the file when a run
  asks for them.
- `test_mp4_output_sink.py` covers the sink that writes an MP4, reading each file
  back to check what was encoded. Its encoding tests are skipped when `ffmpeg` is
  missing from `PATH`.
- `test_client_window_factory.py` covers each way of watching a run answering for
  itself: the window its arguments ask for, the usage error when they are
  incomplete, and what it says about where the output went. The WebRTC tests are
  skipped when the serving packages are missing, which is also why a run writing
  a file does not import them.
- [`apps/t2v/tests`](https://github.com/NVIDIA/flashdreams/tree/main/apps/t2v/tests) covers the reusable text-to-video
  application, session, model loop, and stand-in model checks. Its
  [`test_cli.py`](https://github.com/NVIDIA/flashdreams/blob/main/apps/t2v/tests/test_cli.py) also covers
  `flashdreams-run-v2` itself: finding an application, splitting the command
  line at `--`, choosing a window, describing the session to ask for, and
  running one into a real MP4 with a stand-in for a model. An application that
  describes no session of its own is run there too, since running more than
  text-to-video is the point of the command.
- `test_metrics_output_sink.py` covers the sink that records what a run
  measured, which is a file another tool reads: what a benchmark expects of it
  is checked against the reader itself in
  `flashdreams/tests/test_benchmark_harness.py`.
- The remaining CPU modules cover event buffering, input timelines, native and
  WebRTC windows, UI rendering/compositing, presentation traces, and recent
  frame-rate tracking.

`test_presentation_cuda.py` and `test_webrtc_client_window_cuda.py` are marked
`ci_gpu` and require CUDA. The CPU command below excludes them.

Reusable apps keep their tests beside the package, in `apps/<name>/tests/`
(see `apps/t2v/tests/` above, and `apps/interactive_drive/tests/`).

Run commands from the repository root.

<a id="flashdreams-testv2-readme--set-up-the-test-environment"></a>

## Set up the test environment

```bash

uv sync --package flashdreams-color-fade --package flashdreams-red-screen --package flashdreams-null-model --group test --inexact

```

`test_client_window.py` imports the NULL model integration. The other two small
integrations are included so their v2 tests can use the same environment. `--inexact`
matters: without it, `uv` makes the environment exact for the packages it was
given and uninstalls the rest. `pytest` comes from the `test` group; do not use
`--extra dev`, which pulls `transformer-engine` and compiles CUDA extensions from
source.

<a id="flashdreams-testv2-readme--run-the-tests"></a>

## Run the tests

```bash

uv run --no-sync pytest flashdreams/test_v2 -m ci_cpu -v

```

A single test:

```bash

uv run --no-sync pytest flashdreams/test_v2/test_session_runner.py -v

```

`--no-sync` keeps the run from re-resolving the environment.

The `ci_cpu` tests need no GPU and no model checkpoint. Run the two `ci_gpu`
modules only in a CUDA-capable environment.
