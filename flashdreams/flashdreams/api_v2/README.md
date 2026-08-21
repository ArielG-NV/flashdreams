<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams v2 API

`IApplication` creates one `ISession` for each run. The session implements the
primary model-generation `step`. Session may register independent `IThread` user-visible-threads for tasks such as
UI, game logic, or other stateful work.

## Ownership

- `IApplication` lasts for the process and holds state shared by sessions.
- `ISession` owns state for one run. Its `step` and `reset` methods execute on
  reserved user-visible-thread ID `0`.
- `ISession` defines no constructor, so application sessions keep complete
  control of their own construction. Its user-visible-thread registry initializes lazily.
- User Invisible Threads
    - main-program-thread:
       - Runs session via `run_session` through provided `create_app` method.
      - Launches all other threads.
      - Coordinates io thread initialization with user-visible-thread initialization.
      - Handles rejoining threads launched with rest of the demo to ensure signals to end the program work as intended. This is a spin-loop waiting for threads to rejoin.
    - io-thread:
      - This thread is the first thread launched by the main-program-thread. Logic contained in `run_session::run_io`..
      - Handles initial client-window (WebRTC, Native,...) launch, event collection from client-window.
      - Ticks at a rate of `SessionDesc::frames_per_second_for_ui`.
      - Composites all presentation-ready frames from each backend into a `presentation_buffer` which drains into client-window as often as window permits (adhering to back-pressure). `run_session::when_full` controls the policy to handle a new frame being generated when `presentation_buffer` is full. Composites for presentation onto the client-window backbuffer such-that frame zero is the bottom layer, then frame one, and so on.
- User Visible Threads:
  - model-generation-thread
    - Is an `IThread` that launches implicitly for `ISession` author. `IThread::step` == `ISession::step`, `IThread::reset` == `ISession::reset`.
    - Associated `IThread::state` is the `ISession` that manages your current program session. This thread "owns" your `ISession`.
    - `thread-id` is aquired via `self.get_model_generation_thread()`, Thread-id is 0.
    - Thread is launched implicitly for the user
    - Runs `step` (and therefore presentation) at a tick rate of `frames_per_second_for_step`.
  - All other user-visible-threads
    - Register thread to launch with `ISession::register_thread(thread, thread_id)` inside `ISession::init` via `register_thread`. Trying to register in a different portion of `ISession` will trigger an exception.
    - Tick rate of thread is set by `IThread::frequency`.
    - Thread is implemented via providing a `IThread` or derivative like `UIThread`/ImGUIThread.
    - Threads communicate via `IThread::invoke_async(thread_id, lambda state: ...)`, where `state` is filled with a reference to `IThread::state` (typed via `IThread::StateT`) of the specified `thread_id`. `IThread::invoke_async` adds a message to the `message_queue` of the thread named `thread_id`. `message_queue` for a thread is snapshotted and then processes the snapshot before its next `step` method starts.
    - Fetch a threads last-presented-frame via `IThread::get_last_presented_frame`. Useful for drawing the thread's latest frame in a UI thread (`ImGUIThread::draw_frame`).

## Step-By-Step Of Our Threading Model Lifecycle

**In main-program-thread**
Call `ISession.init`
Start io-thread
[wait for io-thread to launch client-window]
Start user-visible-threads in order from smallest to greatest thread-id
[spin-lock wait for threads to rejoin]
Stop all user-visible-threads
Stop io-thread
Call `ISession.close`

**In io-thread**
Launch client-window
Read client-window input into event_buffer
[wait for user-visible-threads to launch]
[loop with tick rate of `SessionDesc::frames_per_second_for_ui`]
Read client-window input into event_buffer
Collect event_buffer garbage
Compose next presentable frame out of all user-visible-threads
Add frame to presentation_buffer using `when_full` policy.
Drain/present from presentation_buffer

**In any user-visible-thread**
[On loop]
Snapshot message_queue & process the snapshot
Read from event_buffer new user events
End thread if we see a `program-close` user event
If ISesion triggered a `reset`, call `IThread::reset`.
Ensure `IThread` adheres to its tick rate of `IThread::frequency`
End thread if `program-close` user event was set
Run `IThread::step`
Store the last-generated-step & set the last-presented-frame for fetching for 'presentation'/compositing.

## Using ISession user-visible-threads

A user-visible-thread supplies only its state, frequency, and mechanics:

```python
from dataclasses import dataclass

from flashdreams.api_v2.thread import IThread


# This is the ID of the game thread.
GAME_THREAD_ID = 1

# GameState is the state of the game thread.
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

class MySession(ISession):
  def init(self) -> None:
    self.register_thread(GameThread(), GAME_THREAD_ID)
    ...
  
  def step(self, step_index, events) -> StepResult:
    # Increment the game thread's score via message from model-generation-thread.
    invoke_async(GAME_THREAD_ID, lambda state: state.score += 1)
    ...
```

`frequency` is a required non-negative integer giving the maximum number of step
starts per second. Zero means unbounded. Each user-visible-thread has its own `step_index`,
which returns to zero after reset. `SessionDesc.frames_per_second_for_step`
supplies this value only for the model-generation-thread; every user-visible-thread
supplies its own value when constructed.

`IThread.invoke_async` puts a fire-and-forget `Message` in the target user-visible-thread's
thread-safe queue. The operation receives the user-visible-thread's state, must return
`None`, and runs before the next `step`/`step_ui` of that user-visible-thread:

```python
self.invoke_async(game_thread_id, lambda state: state.reset_score())
```

An operation that raises or returns a value other than `None`, fails the user-visible-thread
and triggers shut-down of the entire session. Message Queue operations that have not started when the session stops are discarded.

`IThread.get_model_generation_thread_id` returns the reserved model-generation-thread ID. `IThread.get_last_presented_frame` returns a shared, read-only
`[C, H, W]` tensor. `None` is returned if the target user-visible-thread has contributed a frame to the current generation. Unknown user-visible-thread IDs raise an Exception.

## Input events

The I/O thread appends window input to one arrival-ordered buffer. Every user-visible-thread
has an independent cursor and receives each retained event once. Periodic garbage
collection removes the prefix every active user-visible-thread has read.

A close event requests session-wide shutdown immediately. A reset event advances
the session generation. Every user-visible-thread resets before its next step, and any result
that was still being produced for the previous generation is not presented.

## UI user-visible-threads and compositing

`UIThread` implements `step` for the user. A subclass implements `step_ui`, which
returns one frame, and `wait_for_ui_to_render`, which returns that frame after any
required rendering synchronization.

`flashdreams.runtime_v2.imgui_thread.ImGUIThread` uses
`SlangPyImGUIRenderer`, so an application subclass only supplies the render
dimensions and implements `draw_ui` and any application-state reset.
The renderer owns the ImGui context, external-memory buffer, and synchronization
details on the UI user-visible-thread. `ImGUIThread.draw_frame` places a normalized
`[C, H, W]` video frame inside the active ImGui layout.

Each user-visible-thread publishes its `latest_step` independently. At every I/O tick, the
runtime snapshots the latest current-generation result from each user-visible-thread, selects
its latest frame, and applies its `PresentationMode`:

- `showPresentation` updates the user-visible-thread's last-presented frame and composites it
  into the client backbuffer.
- `hidePresentation` updates the last-presented frame without affecting the
  client backbuffer.
- `disablePresentation` skips presentation and updating the last-presented frame.

Visible frames composite by ascending thread ID: ID `0` is the bottom layer,
then ID `1`, and so on. RGB layers are opaque; RGBA layers use their alpha
channel. Compositing follows `frames_per_second_for_ui` even while main
generation has no new frame.

For sessions with auxiliary workers, the compositor emits one frame in the
session's declared layout, egarly compositing.
For sessions with **only** a model-generation-thread, all processed steps are deterministically forwardwarded to the presentation_buffer. This is a lossless path for MP4 output and benchmarking.
`WhenFull.BLOCK` applies back-pressure while presenting it, and
`WhenFull.DROP_OLDEST` replaces its oldest pending composite. Neither policy
paces or drops an individual user-visible-thread's rendering. The output sink remains a
synchronous consumer and needs no capacity API.

