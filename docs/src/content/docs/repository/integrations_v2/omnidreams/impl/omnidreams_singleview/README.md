---
title: 'OmniDreams Single-View Native'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

<a id="integrationsv2-omnidreams-impl-omnidreamssingleview-readme--omnidreams-single-view-native"></a>

<a id="integrationsv2-omnidreams-impl-omnidreamssingleview-readme--integration-links"></a>

## Integration links

- **Applications:** [omnidreams applications](../../README.md#integrationsv2-omnidreams-readme--integration-links)
- **Configuration:** [omnidreams configuration](../../README.md#integrationsv2-omnidreams-readme--integration-links)

This directory contains the Python helpers and CUDA/C++ extension sources for
the OmniDreams single-view native DiT and LightVAE acceleration path. It also
contains the pinned third-party source manifest, patches, and synchronization
and build helpers used by that extension.

The native loader downloads the manifest's pinned dependencies to
`artifacts/omnidreams/thirdparty/` and compiles the extension on first use.
These sources ship only in a source checkout, not in the OmniDreams wheel. Use
the registered `interactive-drive-omnidreams-perf` application rather than
importing this internal implementation directly; see the
[OmniDreams model guide](../../../../../models/omnidreams.md)
for requirements and launch instructions.
