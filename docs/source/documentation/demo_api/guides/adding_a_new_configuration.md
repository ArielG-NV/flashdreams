---
title: 'Adding a new configuration'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

After [integrating a model adapter for a demo](integrate_model_adapter_for_a_demo.md), you can
add another preset without writing another app. This example adds a faster
pipeline config to the T2V adapter.

## Files to update

```text
integrations_v2/<model>/
  config.py           # define the pipeline variant
  apps/t2v/adapter.py  # add its application factory
  pyproject.toml      # register its CLI name

docs/source/models/<model>.md  # add the preset and run command
```

## Define the pipeline config

In `config.py`, derive a variant from the existing config:

```python
from flashdreams.infra.config import derive_config

PIPELINE_MY_MODEL_FAST = derive_config(
    PIPELINE_MY_MODEL,
    name="my-model-fast",
    diffusion_model={"scheduler": {"num_inference_steps": 4}},
)
```

Choose a step count your model supports. `derive_config` copies the original
config and applies nested changes; an unknown field raises `KeyError`.
Give the variant its own name and add it to the package's config dictionary:

```python
MY_MODEL_CONFIGS = {
    config.name: config
    for config in (PIPELINE_MY_MODEL, PIPELINE_MY_MODEL_FAST)
}
```

The dictionary makes configs available to Python callers. Register a separate
application name to make the variant available from the CLI.

## Add a factory

In `apps/t2v/adapter.py`, reuse the defaults from the previous guide:

```python
from dataclasses import replace

from my_model.config import PIPELINE_MY_MODEL_FAST


def create_app_fast():
    return T2VApplication(
        defaults=replace(
            MY_MODEL_T2V_DEFAULTS,
            pipeline_config=PIPELINE_MY_MODEL_FAST,
        )
    )
```

Like `create_app`, this factory takes no arguments and returns an uninitialized
application. Model loading happens in the app's `init()`.

## Register and run the preset

Add the new name to the existing entry-point table in `pyproject.toml`:

```toml
[project.entry-points."flashdreams.applications_v2"]
"t2v-my-model" = "my_model.apps.t2v.adapter:create_app"
"t2v-my-model-fast" = "my_model.apps.t2v.adapter:create_app_fast"
```

Each CLI name selects a factory, not a key in `MY_MODEL_CONFIGS`.
Reinstall the package to update its entry points:

```bash
uv sync --package flashdreams-<model> --inexact
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 t2v-my-model-fast --mode native-window
```

If a name is missing from `--help`, check that the model package is installed
and the name is registered under `flashdreams.applications_v2`.
For local experiments, the CLI can also import a module with a `create_app()`
function by replacing hyphens in the supplied name with underscores. These
module names are not listed in `--help`; use entry points for published presets.

## Document the preset

Add the new name, its differences, and a run command to
`docs/source/models/<model>.md`. Link to the demo page for shared controls and
arguments. Use a unique name for each preset and keep existing names unchanged.
