---
title: 'Integrate model adapter for a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

An adapter connects a model to an existing demo. It supplies the model settings
and registers a name for `flashdreams-run-v2`.

## Overview

This example connects a model pipeline config to the T2V app. The app handles
the prompt, sessions, and UI; the adapter selects the model and output defaults.

<a id="1-choose-the-existing-interaction"></a>

## Choose a demo

Pick an app from the [demo gallery](../../../demos/index.md) and check its
inputs. T2V accepts a pipeline config; V2V accepts a processor config.
If you need a new interaction, [create a demo](create_demo.md) first.

## Files to add or update

```text
integrations_v2/<model>/
  apps/t2v/
    __init__.py       # add an empty file
    adapter.py        # add the application factory
  pyproject.toml     # add the app dependency and CLI name

docs/source/models/<model>.md  # add a quick start and presets
docs/source/demos/t2v.md       # link to the model's quick start
```

## Add the adapter

In `apps/t2v/adapter.py`:

```python
from t2v import T2VApplication, T2VApplicationDefaults

from my_model.config import PIPELINE_MY_MODEL

MY_MODEL_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_MY_MODEL,
    total_blocks=8,
    pixel_width=832,
    pixel_height=480,
    fps=16,
)


def create_app():
    return T2VApplication(defaults=MY_MODEL_T2V_DEFAULTS)
```

Set the dimensions, frame rate, and block count for your model.
The factory returns an uninitialized app. Load weights and initialize CUDA in
`IApplication.init()`, so listing apps and showing help do not load a model.

For V2V, supply `V2VApplicationDefaults` with a processor, first and subsequent
chunk sizes, and a model name. See the
[SwiftVR adapter](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/swiftvr/apps/v2v/adapter.py)
or [FlashVSR adapter](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/flashvsr/apps/v2v/adapter.py).

## Register the CLI name

Add `flashdreams-t2v` to the integration's existing dependencies and workspace
sources in `pyproject.toml`:

```toml
# Add to [project].dependencies:
# "flashdreams-t2v"

[tool.uv.sources]
flashdreams-t2v = { workspace = true }

[project.entry-points."flashdreams.applications_v2"]
"t2v-my-model" = "my_model.apps.t2v.adapter:create_app"
```

If the package lists modules explicitly in `[tool.setuptools].packages`,
include `<model>.apps` and `<model>.apps.t2v`.

The name `t2v-my-model` is the application slug. Use a lowercase name such as
`<demo>-<model>`. Each entry points to a factory callable without arguments.
Use `flashdreams.applications_v2` for apps; `flashdreams.postprocess_presets`
registers processors separately.

## Install and run

```bash
uv sync --package flashdreams-<model> --inexact
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 t2v-my-model --mode native-window
```

`--help` lists installed application names. For this T2V app's arguments, run
`flashdreams-run-v2 t2v-my-model -- --help`.
Runtime options go before `--`; demo options go after it.

<a id="5-document-once"></a>

## Document the adapter

Add installation instructions, a run command, and the registered preset to
`docs/source/models/<model>.md`. Link to the demo page for controls and
arguments. Add a link back to the model's quick start on the demo page.
For a new model, add it to `docs/source/models/index.md` and
`docs/source/models/.nav.yml` too.

<a id="keep-ownership-one-way"></a>

The app stays in `apps/<demo>/`. Model code and pipeline configs stay in
`integrations_v2/<model>/`; the adapter imports both. Browser and native-window
support come from the runtime. No model-specific copy of the app or `runner.py`
is needed.

Continue with [Adding a new configuration](adding_a_new_configuration.md) to add another
preset to this adapter, or [Interactive serving](interactive_serving.md) to
run it in a browser.
