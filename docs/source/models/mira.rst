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

The example ``mira-mini-1p`` manifest entry uses the 1B single-player bundle from
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

   uv run flashdreams-run --no-instantiate mira \
       --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
       --demo mira-mini-1p

Run the default forward-and-steer sequence:

.. code-block:: bash

   uv run flashdreams-run mira \
       --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
       --demo mira-mini-1p

The runner writes ``mira.mp4`` and ``stats_mira.json`` under
``artifacts/mira/`` by default. Controls use
``KEY+KEY@100MS`` segments, where ``@N`` holds the listed keys for
``N * 100 ms``:

.. code-block:: bash

   uv run flashdreams-run mira \
       --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
       --demo mira-mini-1p \
       --action-script 'W@10,W+D@6,Space@2,W+A@6'

For multiplayer demos, the action script controls player 1 and leaves all
other players inactive. The output MP4 tiles every configured player view in
a compact grid.

Interactive WebRTC UI
---------------------

The MIRA integration includes a browser controller built on the shared
FlashDreams WebRTC backend. Start it with:

.. code-block:: bash

   uv run mira-webrtc \
       --manifest integrations/mira/mira_integration/configs/mira_car_soccer.yaml \
       --demo mira-mini-4p \
       --host 0.0.0.0 --port 8083

Each browser first receives a tiled preview containing every configured player
view. Selecting an available player claims only that player's input stream;
other browsers can use the same URL to claim remaining seats. WebRTC model
definitions in ``mira_integration/configs/mira_car_soccer.yaml`` provide the
checkpoint, player count, and browser-to-checkpoint key map used to construct
the native pipeline, preview, and controls dynamically. Both ``--manifest``
and ``--demo`` are required; MIRA does not select either implicitly.

Open the printed ``/request_session`` URL and select **Start session**. The UI
supports keyboard and touch controls for driving (``WASD``), air roll
(``Q``/``E``), jump (``Space``), boost (``Shift``), and powerslide
(``Control``). The server accepts one active browser session and resets MIRA's
autoregressive cache for each connection.

The MIRA Mini runner is single-GPU. See ``integrations/mira/README.md`` for
the programmatic pipeline API and action vocabulary.
