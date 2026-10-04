---
title: 'Add a v2 model integration'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This guide is the short design checklist. The [v2 integration reference](../repository/integrations_v2/README.md) is the canonical package, application,
testing, and command guide.

## Choose the boundary

Read the [API overview](../api/index.md) before choosing a session contract or
registration group. This page covers applications implementing
`flashdreams.api_v2`, run by `flashdreams.runtime_v2`. The experimental
inference and demo APIs are separate.

## Reuse an application

Prefer an existing reusable application: [T2V](../repository/apps/t2v/README.md),
[Cam2V](../repository/apps/cam2v/README.md),
[Action2V](../repository/apps/action2v/README.md), or
[V2V](../repository/apps/v2v/README.md). The application owns input, session,
loop, UI, and presentation behavior. The integration owns the model pipeline
and a small adapter that supplies application defaults.

Only implement new `IApplication`, `ISession`, or `IModelLoop` classes
when no shared application matches the interaction. See the
[v2 application API](../api/application_api.md) for those contracts.

## Keep model code in its package

Use the current integration shape:

```text

integrations_v2/<model>/
  pyproject.toml
  __init__.py
  config.py
  impl/
  tests/
  apps/<application>/
    __init__.py
    adapter.py

```

`config.py` exposes stable pipeline-config literals. Keep the rest of the
model implementation under `impl/` and model tests under `tests/`. Derive
variants with `flashdreams.infra.config.derive_config` instead of mutating
a shared config.

The adapter exports zero-argument factories that return uninitialized
applications. Expensive checkpoint or CUDA setup belongs in application or
session initialization, never in the factory or at module import time.

## Register and inspect the application

Register each public factory in the integration's `pyproject.toml`:

```toml

[project.entry-points."flashdreams.applications_v2"]
"t2v-customized-method" = "customized_method.apps.t2v.adapter:create_app"

```

Use `<application>-<model>` for the default slug and append descriptive
suffixes for compatible variants. Install and inspect an in-tree package with:

```bash

uv sync --package flashdreams-customized-method --inexact
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 t2v-customized-method -- --help

```

Runtime arguments precede `--`; application arguments follow it. See
[Application slugs](runner_slugs.md) for discovery and ownership details and
[the CLI reference](../api/cli.md) for runtime options.

## Verify and document

Keep import, config, entry-point, and stand-in application tests on CPU. Mark
checkpoint downloads and real model execution `ci_gpu` or `manual`. Every
test must carry a `ci_cpu`, `ci_gpu`, or `manual` marker.

Add the model to [/models/index](../models/index.md), keep its model card focused on user-facing
requirements and the canonical launch command, and link detailed package or
application behavior to the corresponding page under
[/repository/integrations_v2/README](../repository/integrations_v2/README.md).
