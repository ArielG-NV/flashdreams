<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams v2 API

`IApplication` creates one `ISession` for each run. The session implements the
main generation `step` and may register independent `IThread` workers for UI,
game logic, or other stateful work.

## Ownership

- `IApplication` lasts for the process and holds state shared by sessions.
- `ISession` owns state for one run. Its `step` and `reset` methods execute on
  reserved worker ID `0`.
- `ISession` defines no constructor, so application sessions keep complete
  control of their own construction. Its worker registry initializes lazily.
- An auxiliary worker owns its typed `state`. Other threads mutate that state by
  calling `ISession.invoke_async(thread_id, operation)`.
- The runtime owns native threads, event fan-out, frame compositing, and the
  `IClientWindow`.
- Only the I/O thread opens, reads, writes, or closes the client window.

## Session workers

An auxiliary worker supplies only its state, frequency, and mechanics:

```python
from dataclasses import dataclass

from flashdreams.api_v2.thread import IThread


@dataclass
class GameState:
    score: int = 0


class GameThread(IThread[GameState]):
    def __init__(self) -> None:
        super().__init__(state=GameState(), frequency=60)

    def step(self, step_index, events):
        ...

    def reset(self) -> None:
        self.state.score = 0
```

`frequency` is a required non-negative integer giving the maximum number of step
starts per second. Zero means unbounded. Each worker has its own `step_index`,
which returns to zero after reset. `SessionDesc.frames_per_second_for_step`
supplies this value only for the main-generation worker; every auxiliary worker
supplies its own value when constructed.

`invoke_async` puts a fire-and-forget `Message` in the worker's thread-safe
queue. The operation receives the worker-owned state, must return `None`, and
runs before the next `step`/`step_ui` of that thread:

```python
session.invoke_async(game_thread_id, lambda state: state.reset_score())
```

An operation that raises, or returns a value other than `None`, fails the worker
and shuts down the session. Queued operations that have not started when the
session stops are discarded.

## Input events

The I/O thread appends window input to one arrival-ordered buffer. Every worker
has an independent cursor and receives each retained event once. Periodic garbage
collection removes the prefix every active worker has read.

A close event requests session-wide shutdown immediately. A reset event advances
the session generation. Every worker resets before its next step, and any result
that was still being produced for the previous generation is not presented.

## UI workers and compositing

`UIThread` implements `step` for the user. A subclass implements `step_ui`, which
returns one frame, and `wait_for_ui_to_render`, which returns that frame after any
required rendering synchronization.

`flashdreams.runtime_v2.imgui_thread.ImGUIThread` uses
`SlangPyImGUIRenderer`, so an application subclass only supplies the render
dimensions and implements `draw_ui` and any application-state reset.
The renderer owns the ImGui context, external-memory buffer, and synchronization
details on the UI worker.

Each worker publishes its `latest_step` independently. At every I/O tick, the
runtime snapshots the latest current-generation result from each worker, selects
its latest frame, and composites enabled frames by ascending thread ID: ID `0`
is the bottom layer, then ID `1`, and so on. RGB layers are opaque; RGBA layers
use their alpha channel. `StepResult.disabled` hides that worker's layer.
Compositing follows `frames_per_second_for_ui` even while main generation has no
new frame.

The compositor emits one frame in the session's declared layout. Only this
composited UI presentation stream is bounded by `max_pending`.
`WhenFull.BLOCK` applies back-pressure while presenting it, and
`WhenFull.DROP_OLDEST` replaces its oldest pending composite. Neither policy
paces or drops an individual worker's rendering. The output sink remains a
synchronous consumer and needs no capacity API.

## Lifecycle

`run_session` performs this sequence:

1. Register the main generation adapter as thread `0`.
2. Call `ISession.init`, where auxiliary workers may be registered.
3. Freeze registration, open the window, and collect initial input.
4. Start every worker and run the I/O loop.
5. On close, main-step completion, or failure, request shutdown.
6. Wait for in-flight worker steps, discard unprocessed messages, close the
   window, and call `ISession.close`.

`steps=N` counts completed main-generation steps across resets. Auxiliary workers
continue until the main worker reaches that count. `steps=None` runs until a
close event or failure.
