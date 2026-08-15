<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Application and runner slugs

The short public name after `flashdreams-run` can select an application or a
runner. An exact installed application slug is routed to the shared application
host first; otherwise the name selects a registered `RunnerConfig`.

## Quick start

After installing the OmniDreams workspace package, write the default replay to
MP4 through the application host:

```bash
uv sync --package flashdreams-omnidreams
uv run flashdreams-run omnidreams --output mp4 --example-data
```

The package also registers the closed-loop application:

```bash
uv run flashdreams-run interactive-drive
```

The shipped public application slugs are `omnidreams`, `omnidreams-perf`, and
`interactive-drive`. The two replay applications derive their pipeline defaults
from the corresponding runner configs; the runner entries remain available as
configuration identities and for non-demo runner workflows.

## How an application command is dispatched

For this command:

```bash
uv run flashdreams-run omnidreams --output mp4 --example-data
```

the control flow is:

```text
flashdreams-run console script
  -> flashdreams.scripts.cli:entrypoint
  -> registered_application_slugs()
  -> flashdreams.demo.application:entrypoint
  -> create_application("omnidreams")
       -> flashdreams.applications entry point
       -> omnidreams.demo.application:create_app
  -> application.init(["--example-data"])
  -> application.create_session()
  -> shared host loop with Mp4IOFactory
```

Application slugs take precedence over runner slugs on an exact match. The host
consumes transport flags such as `--output`, `--output-path`, `--host`, and
`--port`; all remaining arguments are passed to the application. This keeps
model/session logic independent of local-window, MP4, null, and WebRTC I/O.

The reusable driving contract lives in
`apps/interactive_drive/interactive_drive.py`;
OmniDreams supplies its runner and factory without importing model code into the
shared package. Application factories are registered separately from runner
configs:

```toml
[project.entry-points."flashdreams.applications"]
"interactive-drive" = "omnidreams.interactive_drive.runner:create_app"
"omnidreams" = "omnidreams.demo.application:create_app"
"omnidreams-perf" = "omnidreams.demo.application:create_perf_app"
```

## How a runner slug is registered

External integrations register runners with Python package entry points. The
OmniDreams package declares:

```toml
[project.entry-points."flashdreams.runner_configs"]
"omnidreams" = "omnidreams.config:RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE"
"omnidreams-perf" = "omnidreams.config:RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF"
```

Each target resolves to an `OmnidreamsRunnerConfig` literal:

```python
RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE = OmnidreamsRunnerConfig(
    runner_name="omnidreams",
    pipeline=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    # ...
)
```

Discovery loads every entry point in the `flashdreams.runner_configs` group.
The registry key comes from `cfg.runner_name`, not from the entry-point name.
The entry-point name should still match `runner_name`: keeping both aligned
makes installed package metadata understandable, and the OmniDreams CPU tests
enforce that invariant.

Slug collisions are deterministic. Built-in runners take precedence over
plugins, and the first discovered plugin with a given `runner_name` wins over
later plugins. The registry logs skipped collisions.

## Adding or changing a public slug

1. Choose a short, integration-level default such as `omnidreams`. Add a
   concise suffix only when users need to select a materially different public
   variant, such as `omnidreams-perf`.
2. Set `runner_name` on the exported runner config literal.
3. Add an entry with the same name under
   `[project.entry-points."flashdreams.runner_configs"]`.
4. Keep `pipeline.name` unchanged unless the actual model recipe identity is
   changing.
5. Update launch manifests, CI commands, benchmark scenarios, and user-facing
   documentation. Do not rename checkpoint keys, asset paths, reference
   artifacts, or internal preset IDs merely because the public slug changed.
6. Refresh the editable package metadata and run CPU-only checks:

```bash
uv sync --package flashdreams-omnidreams --package flashdreams-lingbot \
  --group test
uv run flashdreams-run --help
uv run flashdreams-run omnidreams --help
uv run pytest -m ci_cpu \
  integrations/omnidreams/tests/test_recipe_configs.py \
  integrations/omnidreams/tests/test_demo_api.py \
  flashdreams/tests/test_launch.py \
  flashdreams/tests/test_launch_manifest.py
```

The `--help` check resolves application registration and parsing without
loading checkpoints or initializing the GPU.
