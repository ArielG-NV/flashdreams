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

FlashDreams
===================================

.. raw:: html

   <style>
     #furo-main-content > section > h1 { display: none; }
   </style>
   <div class="homepage-logo-wrap">
     <img src="_static/flashdreams-logo-horizontal.png" alt="FlashDreams">
   </div>

High-performance inference and serving for interactive autoregressive world
models.

FlashDreams turns video and world models into responsive systems. Instead of
offline prompt-in/video-out jobs, it keeps a live loop open:
**input -> encode -> model step -> streamed output -> next input**.

.. raw:: html

   <div class="fd-stat-grid">
     <div class="fd-stat-card">
       <div class="fd-stat-value">2.62x</div>
       <div class="fd-stat-label">Lingbot-World speedup vs official DiT path</div>
     </div>
     <div class="fd-stat-card">
       <div class="fd-stat-value">2.12x</div>
       <div class="fd-stat-label">Self-Forcing speedup vs FastVideo</div>
     </div>
     <div class="fd-stat-card">
       <div class="fd-stat-value">1.40x</div>
       <div class="fd-stat-label">Wan2.1 speedup vs FastVideo</div>
     </div>
     <div class="fd-stat-card">
       <div class="fd-stat-value">8</div>
       <div class="fd-stat-label">Integrated model families in one runtime</div>
     </div>
   </div>

Why FlashDreams
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Realtime autoregressive inference

      Cache-aware long rollouts.

   .. grid-item-card:: Interactive serving backend

      Persistent sessions with streamed output.

   .. grid-item-card:: Multi-GPU scaling

      Context-parallel execution through ``torchrun``.

   .. grid-item-card:: Extensible model ecosystem

      Built-in and standalone model integrations.

Performance Highlights
----------------------

Benchmarks on popular open model families show strong gains against official
implementations and widely used video inference libraries.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Lingbot-World

      Up to **2.62x** faster than the official implementation and **1.60x**
      faster than LightX2V in matched DiT-only measurements.

   .. grid-item-card:: Self-Forcing

      Up to **2.12x** faster than FastVideo on GB300 for the 6th
      autoregressive block.

   .. grid-item-card:: Wan2.1

      Up to **1.40x** faster than FastVideo for 480p, 81-frame DiT inference
      with CFG.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Lingbot-World

      .. image:: /_static/perf/perf-0521-lingbot-world.svg
         :alt: Lingbot-World benchmark chart.

   .. grid-item-card:: Self-Forcing

      .. image:: /_static/perf/perf-0521-self-forcing.svg
         :alt: Self-Forcing benchmark chart.

   .. grid-item-card:: Wan2.1

      .. image:: /_static/perf/perf-0521-wan21.svg
         :alt: Wan2.1 benchmark chart.

Serving Showcase
----------------

.. raw:: html

   <div class="fd-highlight-grid">
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Lingbot-World</div>
       <div class="fd-highlight-body">Camera-control world-model serving for interactive navigation.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">OmniDreams</div>
       <div class="fd-highlight-body">Closed-loop autonomous-vehicle simulation with realtime model feedback.</div>
     </div>
   </div>

- :doc:`Lingbot-World details </models/lingbot_world>` and
  `project page <https://technology.robbyant.com/lingbot-world>`_.
- :doc:`OmniDreams details </models/omnidreams>` and
  `blog page <https://research.nvidia.com/labs/sil/projects/omnidreams-blog/>`_.

To understand why this is different from offline video generation, start with
:doc:`/getting_started/offline_vs_online`.

Best For
--------

.. raw:: html

   <div class="fd-pill-row">
     <span class="fd-pill">World-model researchers</span>
     <span class="fd-pill">Video generation teams</span>
     <span class="fd-pill">Simulation platforms</span>
     <span class="fd-pill">Robotics</span>
     <span class="fd-pill">Autonomous vehicles</span>
     <span class="fd-pill">Healthcare workflows</span>
     <span class="fd-pill">Creative tools</span>
     <span class="fd-pill">Virtual environments</span>
   </div>

Integrated Models
-----------------

.. raw:: html

   <div class="fd-pill-row">
     <span class="fd-pill">OmniDreams</span>
     <span class="fd-pill">Self-Forcing</span>
     <span class="fd-pill">Causal-Forcing</span>
     <span class="fd-pill">Causal-Wan2.2</span>
     <span class="fd-pill">Lingbot-World</span>
     <span class="fd-pill">FlashVSR</span>
     <span class="fd-pill">Cosmos-Predict2.5</span>
     <span class="fd-pill">Wan2.1</span>
   </div>

See :doc:`/models/index` for commands, variants, and upstream links.

How It Gets Fast
----------------

LLM runtimes optimize token prefill/decode. Video libraries optimize offline
generation. FlashDreams optimizes persistent world-model loops with cuDNN
attention, CUDA Graph, NVJPEG media flow, and cache-aware execution.

Start Here
----------

Choose the path that matches what you want to do next.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: getting_started/index
      :link-type: doc

      Install FlashDreams, understand online world-model inference, and launch
      your first inference and serving runs.

   .. grid-item-card:: Developer Guides
      :link: developer_guides/index
      :link-type: doc

      Learn the repo architecture, programmatic usage, model integration path,
      config system, and serving design.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      CLI usage and API surfaces.

   .. grid-item-card:: Models
      :link: models/index
      :link-type: doc

      Find per-model installation notes, runner slugs, commands, upstream links,
      and performance notes.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/index
   getting_started/offline_vs_online
   getting_started/installation
   getting_started/first_world_model
   getting_started/supported_models

.. toctree::
   :maxdepth: 1
   :caption: Developer Guides
   :hidden:

   developer_guides/index
   developer_guides/system_overview
   developer_guides/usage_patterns
   developer_guides/configs
   developer_guides/interactive_serving
   developer_guides/new_recipes

.. toctree::
   :maxdepth: 1
   :caption: Models
   :hidden:

   models/index
   models/omnidreams
   models/self_forcing
   models/causal_forcing
   models/fastvideo_wan22
   models/lingbot_world
   models/flashvsr
   models/cosmos_predict2
   models/wan21

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index
   CLI <reference/cli>
   API <apis/index>
