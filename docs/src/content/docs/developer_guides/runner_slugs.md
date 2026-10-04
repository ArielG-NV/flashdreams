---
title: 'Application slugs and model adapters'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This guide covers `flashdreams-run-v2`. Its application protocol and entry
point group are separate from the `flashdreams-run` inference and demo APIs;
see the [API overview](../api/index.md).

## Discovery

Each integration registers zero-argument application factories in
`flashdreams.applications_v2`:

```toml

[project.entry-points."flashdreams.applications_v2"]
"cam2v-lingbot" = "lingbot.apps.cam2v.adapter:create_app"
"cam2v-lingbot-world-fast" = "lingbot.apps.cam2v.adapter:create_app_fast"

```

List the slugs installed in the current environment, then inspect one
application's arguments:

```bash

uv run flashdreams-run-v2 --help
uv run flashdreams-run-v2 cam2v-lingbot -- --help

```

The default slug is normally `<application>-<model>`. Compatible variants
append a descriptive suffix and map to an explicit factory in the same adapter.
The registered entry points in each integration's `pyproject.toml` are the
source of truth.

## Ownership

Reusable application behavior belongs under `apps/<application>/`. Model
implementation, configuration, and tests belong under
`integrations_v2/<model>/`. The only bridge is the small
`apps/<application>/adapter.py` module in the model package.

Do not add `runner.py`, `launch.py`, `runtime.py`, `model_session.py`,
or a model-specific copy of an existing application merely to make a v2 slug.
See [Add a v2 model integration](new_integration.md) for the concise workflow
and [the v2 integration reference](../repository/integrations_v2/README.md) for
the complete package and testing rules.
