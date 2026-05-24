.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

System overview
===================================

FlashDreams keeps shared runtime code separate from model integrations.

.. raw:: html

   <div class="ai-figure-placeholder">
     <div class="ai-figure-title">Figure placeholder: FlashDreams repo and runtime architecture</div>
     <div class="ai-figure-body">
       Replace this block with an AI-generated 16:9 architecture illustration.
       The figure should show the repo layers and runtime flow from runner to
       pipeline to model components.
     </div>
   </div>

.. dropdown:: AI architecture figure prompt

   .. code-block:: text

      Create a beautiful technical architecture illustration for FlashDreams
      developer documentation.

      Show the repo design as layered blocks:
      core: attention, distributed utilities, IO primitives.
      infra: configs, runner, pipeline, encoder, transformer, scheduler, decoder,
      serving contracts.
      recipes: reusable model components.
      integrations: standalone plugin-style model packages.

      Show the conceptual runtime flow across the center: Runner -> Pipeline ->
      Encoder -> Diffusion / Transformer / Scheduler -> Decoder -> Output, with
      an optional Serving Session wrapping the loop for streaming applications.

      Visual style: clean modern developer-doc architecture diagram, vector-like,
      elegant accent colors, readable labels, balanced spacing, soft shadows,
      minimal clutter, white or dark neutral background, professional and
      attractive. The figure should feel like a high-level product architecture
      overview rather than a dense UML diagram. Aspect ratio 16:9, high
      resolution, crisp text.

.. figure:: /_static/diagrams/system_overview_flow.svg
   :alt: High-level FlashDreams system flow from CLI to pipeline and outputs.

   High-level flow across CLI, runner, pipeline, diffusion/model components,
   and distributed runtime.

Repo map
--------

.. raw:: html

   <div class="fd-highlight-grid">
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">core</div>
       <div class="fd-highlight-body">Attention, distributed helpers, checkpoint loading, and IO utilities.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">infra</div>
       <div class="fd-highlight-body">Configs, runner, pipeline, encoder, decoder, scheduler, CUDA graph, and serving contracts.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">recipes</div>
       <div class="fd-highlight-body">Reusable in-package model components shared by integrations.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">integrations</div>
       <div class="fd-highlight-body">Standalone plugin-style model packages with configs and runners.</div>
     </div>
   </div>

Execution flow (code-mapped)
----------------------------

``flashdreams-run`` resolves a runner, the runner prepares user inputs, the
pipeline executes ``initialize_cache`` -> ``generate`` -> ``finalize``, and
rank 0 writes user-facing artifacts.

Code map
--------

- :doc:`/apis/core` for low-level kernels and distributed utilities.
- :doc:`/apis/infra` for pipeline, diffusion, and runner abstractions.
- :doc:`/apis/recipes` for pipeline/runner API surfaces and integration map.
- :doc:`/developer_guides/usage_patterns` for running existing models,
  programmatic access, and choosing the right developer workflow.
- :doc:`/developer_guides/new_recipes` for implementing and registering new
  models.
