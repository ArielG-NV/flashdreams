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

Self-Forcing
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://self-forcing.github.io/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2506.08009" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://github.com/guandeh17/Self-Forcing" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Self-Forcing is a Wan2.1-based text-to-video (T2V) model.
It uses a training paradigm for autoregressive video diffusion that simulates
inference-time rollout during training with KV caching, reducing the train-test
gap and enabling efficient streaming generation quality.

.. image:: https://self-forcing.github.io/static/teaser.jpg
   :alt: Self-Forcing teaser figure.
   :width: 100%

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/self_forcing

Running the method
------------------

To run Self-Forcing, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --prompt "A cat surfing on a neon wave." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``self-forcing-wan2.1-t2v-1.3b``
     - Self-Forcing distilled Wan 2.1 1.3B T2V (Wan VAE decoder, Official).
   * - ``self-forcing-wan2.1-t2v-1.3b-flash``
     - Self-Forcing distilled Wan 2.1 1.3B T2V (Faster TAEHV decoder).
   * - ``self-forcing-wan2.1-t2v-1.3b-anti-drift``
     - Self-Forcing distilled Wan 2.1 1.3B T2V (sink + sliding window, with KV cache re-ROPE).

For multi-GPU inference, simply use ``uv run torchrun --nproc_per_node=4 --no-python flashdreams-run``
instead of ``uv run flashdreams-run`` (taking 4 GPUs as an example).

Performance Comparison
----------------------

.. figure:: /_static/perf/self_forcing_total_ms.svg
   :class: benchmark-figure
   :figclass: benchmark-figure-wrap
   :alt: Self-Forcing total latency bar chart by hardware and method.

   This chart shows the DiT runtime at 6-th autoregressive rollout on a signle GPU.
   Both using CUDNN attention backend. See
   `parity check <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing/tests/parity_check>`_
   for scripts to run profiling on the official implementation.
