---
title: 'Red Screen'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="integrationsv2-redscreen-readme--red-screen"></a>

<a id="integrationsv2-redscreen-readme--integration-links"></a>

## Integration links

- **Applications:** [Red Screen](README.md#integrationsv2-redscreen-readme--run-it)
- **Configuration:** [`red_screen/app.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/red_screen/red_screen/app.py)

Smallest interactive v2 application. It holds no learned model: a session emits
red frames whose intensity is driven by keyboard input in the browser.
`color_fade` is the counterpart that writes a file; this one is about input.

<a id="integrationsv2-redscreen-readme--run-it"></a>

## Run it

```bash

uv sync --package flashdreams-red-screen --inexact
uv run --no-sync red-screen-webrtc

```

Open the URL it prints. Hold `r` to turn the generated video red, `w` to raise
the red intensity by 0.1, and `s` to lower it.

This integration ships its own launcher rather than relying on
`flashdreams-run-v2`, which is why the session arguments are spelled differently
here:

```bash

uv run --no-sync red-screen-webrtc --host 0.0.0.0 --port 8080 \
    --width 1280 --height 720 --fps 30 -- --key x

```

It is also reachable the usual way, through the module fallback in the
application registry:

```bash

uv run --no-sync flashdreams-run-v2 red-screen --mode webrtc

```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--key` | `r` | Key whose held state selects red over black. |

Output is a `[1, 3, 1, H, W]` float32 tensor in `bcthw` layout carrying `[-1, 1]`
values. The green and blue channels stay at `-1.0`. The red channel is `1.0`
while the activation key is held; otherwise it starts at `-1.0` and `w`/`s`
adjust it in 0.2 tensor-value increments. Any other layout is refused.

<a id="integrationsv2-redscreen-readme--what-it-is-worth-reading-it-for"></a>

## What it is worth reading it for

- An application module implements `IApplication`, `ISession`, and `IModelLoop`
  without a custom `IUILoop`. This one never names `InputSource` or `OutputSink`;
  its standalone launcher uses the runtime's `IClientWindow` factory.
- A session is created from a `SessionDesc` before any client connects, and
  reports back what it resolved to.
- Keyboard events are edges, not levels. A key stays held across steps carrying
  no events for it, so a key-down at step 0 keeps the screen red until the
  matching key-up arrives. Anything reading input needs this bookkeeping.
- `reset()` is implemented, so a browser client can start the session over.

Driving it from Python takes the same objects the launcher builds:

```python

app = create_app()
app.init([])
session = app.create_session(
    SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        video_width=16,
        video_height=16,
    )
)
run_session(session, my_client_window, steps=4)
app.close()

```

`steps` bounds the run for a test. An interactive run leaves it `None` and ends
when the browser disconnects.

<a id="integrationsv2-redscreen-readme--tests"></a>

## Tests

```bash

uv sync --package flashdreams-red-screen --group test --inexact
uv run --no-sync pytest integrations_v2/red_screen -m ci_cpu -v

```

Input is driven by `ScriptedClientWindow` in
[`red_screen/tests/test_red_screen.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/red_screen/red_screen/tests/test_red_screen.py), which is
the pattern to copy for anything else that reads events. See the
[integration guide](../README.md) for what `--inexact` is doing there.
