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

MIRA Mini
=========

MIRA Mini is an action-conditioned autoregressive world model for interactive
car soccer. FlashDreams implements the model runtime natively: learned action
conditioning, flow matching, temporal ``BlockKVCache`` state, bootstrap video
encoding, and causal video decoding all use FlashDreams contracts and
primitives. The external ``alakazam-mira-mini`` package is not required.

The default demo uses the 1B single-player bundle from
`alakazamworld/mira-mini <https://huggingface.co/alakazamworld/mira-mini>`_.
Model loading and the roughly 12 GB download are deferred until a rollout is
initialized, so config inspection remains CPU-safe.

Installation
------------

.. code-block:: bash

   uv sync --package flashdreams-mira --extra dev --extra runners
   uv pip install imageio-ffmpeg

The Hugging Face client reads ``HF_TOKEN`` from the environment. The weights
are licensed CC BY-NC-SA 4.0: non-commercial use, attribution, and share-alike
terms apply.

Scripted demo
-------------

Inspect the config without loading weights:

.. code-block:: bash

   uv run flashdreams-run --no-instantiate mira-mini-1b-demo

Run the default forward-and-steer sequence:

.. code-block:: bash

   uv run flashdreams-run mira-mini-1b-demo

The runner writes ``mira-mini-1b-demo.mp4`` and
``stats_mira-mini-1b-demo.json`` under its output directory. Controls use
``KEY+KEY@STEPS`` segments:

.. code-block:: bash

   uv run flashdreams-run mira-mini-1b-demo \
       --action-script 'W@10,W+D@6,Space@2,W+A@6'

Local checkpoints
-----------------

Use an already downloaded bundle by supplying native Windows paths:

.. code-block:: powershell

   uv run flashdreams-run mira-mini-1b-demo `
       --pipeline.bundle-path C:/models/mira-mini `
       --pipeline.checkpoint-path C:/models/mira-mini/checkpoint-52000/checkpoint.pth `
       --pipeline.context-path C:/models/mira-mini/context/default.npz

The MIRA Mini runner is single-GPU. See ``integrations/mira/README.md`` for
the programmatic pipeline API and action vocabulary.
