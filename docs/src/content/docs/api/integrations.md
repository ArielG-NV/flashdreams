---
title: 'Pipelines, runners, and applications'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams model integrations use these layers:

- **Pipelines** (`StreamInferencePipelineConfig`) that define model behavior.
- **Runners** (`RunnerConfig` + `Runner`) that define CLI-facing I/O.
- **V2 applications** (`IApplication`) that bind reusable application
  infrastructure directly to pipeline configs.

Most actively developed model implementations now live under
`integrations_v2/<name>/` as plugin-style standalone packages. This page
keeps documenting the in-tree pipeline modules that are still exposed from
`flashdreams.recipes`.

For the application and loop contracts, see [application_api](application_api.md). For the
other API boundaries and registration groups, see the [API overview](index.md). The [integration layout guide](../repository/integrations_v2/README.md) owns the package contract and complete
integration inventory.

:::note

Pipeline modules import the heavy GPU stack (transformer-engine, CUDA
ops) at import time, so this page shows them by *automodule* with
`:no-undoc-members:` to keep the rendered API focused on the names
that these in-tree modules actually expose.

::: 

## V2 integration packages

A v2 integration registers an `IApplication` factory through the
`flashdreams.applications_v2` entry-point group. Use the
[integration layout guide](../repository/integrations_v2/README.md) for the current directory contract and
per-package implementation notes, and the [model gallery](../models/index.md)
for supported user-facing launch commands.

## Wan

### `flashdreams.recipes.wan`

Public Wan integration surface for integration plugins.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/recipes/wan/__init__.py)

### `flashdreams.recipes.wan.pipeline`

Unified Wan inference pipeline (Wan 2.1 / Wan 2.2, T2V and I2V).

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/recipes/wan/pipeline.py)

## TAEHV

### `flashdreams.recipes.taehv`

TAEHV video decoder and Hunyuan Video 1.5 codec configs.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/recipes/taehv/__init__.py)
