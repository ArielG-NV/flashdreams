<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Protocols for the FlashDreams API.

- `application.py` / `session.py`: `IApplication` creates an `ISession` from a
  `SessionDesc`, and the session reports what it resolved to. `session_desc`
  is the description the application would choose for itself, for a caller with
  none of its own.
- `input_source.py` / `output_sink.py` / `client_window.py`: `IClientWindow` is
  one client's input and output together. It is given the session's `SessionDesc`
  in `OutputSink.open`.
- `user_input_event_data.py`: base type for event payloads.

`flashdreams.runtime_v2.session_runner.run_session` drives a session against a
window until the session reports `is_finished` or the window reports a close, or
for a fixed number of steps a caller asks for. A caller holding an application
uses `flashdreams.runtime_v2.application_runner.ApplicationRunner` to get there,
which takes no step count: how long a run lasts is the application's business. A
run whose output is a file goes the same way, against
`flashdreams.runtime_v2.mp4_client_window.Mp4ClientWindow`, which reports no
input and encodes every result. Since it never reports a close, such a run needs
a session that finishes.

`flashdreams-run-v2` is that run from a shell: `flashdreams.runtime_v2.cli` finds
an application by slug, gives it the arguments after `--`, and hands it to
`ApplicationRunner` with the window `--mode` asked for, an MP4 file or a client
over WebRTC. Applications are found through the `flashdreams.applications_v2`
entry point group, or by the name of the package an integration ships when it
has registered nothing, which is
`flashdreams.runtime_v2.application_registry`'s job.

What the modes are belongs to
`flashdreams.runtime_v2.client_window_factory`, not to the command. A mode owns
the arguments only it takes, such as `--output-path` for a file or `--port` for
a browser, and what to say about where the run went: a URL to open before it
starts, or the file once there is something in it. So a new way of watching a
run is a mode added there, and the command is unchanged.

The session it asks for comes from `IApplication.session_desc`, with
`--pixel-width`, `--pixel-height`, `--fps`, `--ui-fps`, and `--layout`
overriding whatever they name. That is the whole of what the command knows
about the kind of application it is running: a model answers with the clip its
checkpoint was trained for, and an application that generates whatever it is
asked for answers nothing and is described by those arguments alone.

`--stats-path` asks a run to record what it cost as well as what it generated.
`Mp4ClientWindow` takes that path and adds a `MetricsOutputSink` beside the MP4
writer, which records each step's measurements as the artifact
`flashdreams-benchmark` reads. The measurements are the model's own: a step reports what
it measured and this writes it down, converting milliseconds to seconds because
a report cannot compare two units. Nothing is measured unless a run asks, so an
ordinary run pays nothing for this.
[`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
is the suite that uses it, comparing every t2v model on one prompt and seed, and
[running it](../../tools/benchmarks/README.md) is written down beside the
harness.

`flashdreams.t2v_v2` is text-to-video on top of these protocols rather than part
of them: one `T2VApplication` owns the command line every t2v model needs, an
integration supplies only its own defaults, and `testing.check_t2v_model_impl`
is the check its tests run to cover the batch path in one call. See
[its README](../t2v_v2/README.md). The five `integrations_v2/t2v_*` packages are
the models behind it, and each is a factory of about forty lines.

Ownership
---------

Agreed design decisions. Change them by discussion.

- An application module implements `IApplication` and `ISession`. The runtime
  creates every other protocol here and passes it in.
- `IApplication` lasts as long as the process. It holds what its sessions share,
  such as a checkpoint or a compiled pipeline, and outlives every session it
  creates. It also says what session it would generate unasked, through
  `session_desc`, since only it knows what its model was trained for. The
  default says nothing, for an application that generates whatever it is asked
  for.
- `ISession` is one run: KV cache, game state, and anything else that must not
  carry into another run. It also says when that run is over, through
  `is_finished`. The default never finishes.
- `InputSource` and `OutputSink` belong to the runtime. The runner reads from the
  source and writes to the sink, so a session takes `UserInputEvents` in, returns
  a `StepResult`, and holds neither.
- `IClientWindow` pairs one client's input source and output sink. It is internal
  to the runtime, which is why it appears in no signature on `IApplication` or
  `ISession`. A window whose client disconnects reconnects itself rather than the
  runtime creating a second session.
- Application and session logic, including UI rendering, runs on the server side
  and is presented or streamed to a client window.
- The `UserInputEventData` types in `flashdreams.runtime_v2` cover the input
  modalities supported today, and integrations consume them. Nothing stops an
  integration subclassing the base class, and whether it should be able to is not
  settled, so this is a convention rather than something the code enforces.
- Ending and restarting a run are events on that same stream, not separate calls:
  a window reports `CloseUserInputEventData` when its client closes or goes away,
  and `ResetUserInputEventData` to start over. This is how native windowing
  systems deliver a close, ordered with the input around it, and each
  user-visible-thread is handed the batch it arrived in, so a session can react
  rather than just being abandoned.
- A reset does not split the input around it. The batch reaches the first step of
  the new generation whole, so a key held down when the client restarts is still
  held after, because it is the earlier edge that says so. A session that must
  not inherit that input ignores the older events itself.
- An `OutputSink` reads `StepResult.output` as one of two things: floats holding
  `[-1, 1]`, which is what FlashDreams models emit, or integers holding raw
  `0`-`255`. `SessionDesc` carries no range and a session cannot declare one, so
  this is a convention every sink follows.

Threading
---------

`IApplication` creates one `ISession` for each run. During initialization the
session registers one model-generation-thread and may register additional
user-visible-threads for UI, game logic, or other stateful work.

### Thread model

- `IApplication` lasts for the process and holds state shared by sessions.
- `ISession` owns the lifecycle and thread registry for one run.
- The registered model-generation-thread executes on reserved
  user-visible-thread ID `0`.
- `ISession` defines `init` to register threads and `close` to clean-up any auxillary resources initialized in `init`. 

Program Threads:
  - `main-program-thread`:

    - Runs the session through `run_session`, called through the provided
      `create_app` method.
    - Launches all other threads.
    - Coordinates io-thread initialization with user-visible-thread
      initialization.
    - Rejoins the threads launched with the rest of the demo so signals end the
      program as intended. This is a spin loop waiting for threads to rejoin.

  - `io-thread`:

    - Is the first thread launched by the main-program-thread. The logic is in
      `run_session.run_io`.
    - Launches the client window (WebRTC, native, or another implementation) and
      collects events from it.
    - Ticks at `SessionDesc.frames_per_second_for_ui`.
    - Merges presentation-ready frames from each user-visible-thread into a single frame storing result in our `presentation_buffer`. The buffer drains into the client window as often
      as the window permits, subject to back-pressure. `run_session.when_full`
      controls what happens when a frame is generated while the buffer is full.
      Thread ID `0` is the bottom layer, followed by ID `1`, and so on.

- `user-visible-threads`:

  - `model-generation-thread`:

    - Is an `IThread` explicitly registered by `ISession.init` through
      `ISession.register_model_generation_thread`.
    - Reserves thread ID `0` for itself, returned by
      `IThread.get_model_generation_thread_id`.
    - Ticks `step` at
      `SessionDesc.frames_per_second_for_step`.

  - All other user-visible-threads:

    - Register with
      `ISession.register_thread(thread_id, thread_type, state=..., frequency=..., ...)`
      during `ISession.init`. Registering elsewhere raises an exception.
      Arguments depend on the type of thread being registered; all arguments
      from `state` onward are forwarded to the `IThread` implementation's
      `__init__` method.
    - Tick at `IThread.frequency`.
    - Implement `IThread` or a subclass such as `UIThread` or `ImGUIThread`.

  - All user-visible-threads:
    - Communicate through `IThread.invoke_async` from a user-visible-thread to another user-visible-thread:
      `invoke_async(thread_id, lambda state: ...)`, where `state` is the target thread's
      `IThread.state` (typed by `IThread.StateT`). This method adds a message to
      a threads `message_queue`.
    - Message queue processes via: 1. Snapshot the queue, 2. Processing the snapshot, 3. Clearing the processed messages.
    - Fetch a thread's last-presented frame through
      `IThread.get_last_presented_frame`. This can be used with `ImGUIThread.draw_frame` to paint the last-presented frame of
      another user-visible-thread (like the model-generation-thread) onto ImGUI UI elements.

### Lifecycle

`main-program-thread`:

1. Call `ISession.init`, which registers the model-generation-thread and any
   additional user-visible-threads.
2. Start the io-thread and wait for it to open the client window.
3. Start the user-visible-threads.
4. Wait for the io-thread to finish.
5. Stop and join the user-visible-threads, clear `event_buffer`, and call
   `ISession.close`.

`io-thread`:

1. Open the client window and read its input into `event_buffer`.
2. Wait for the user-visible-threads to launch.
3. At each `SessionDesc.frames_per_second_for_ui` tick:

   1. Read client-window input into `event_buffer`.
   2. Collect `event_buffer` garbage.
   3. Compose the next presentable frame from all user-visible-threads.
   4. Add the frame to `presentation_buffer` using the `when_full` policy.
   5. Drain `presentation_buffer` into the client window.

Each `user-visible-thread`:

1. Snapshot `message_queue` and process the snapshot.
2. Read new user events from `event_buffer`.
3. End the thread if it receives a close event.
4. If detected that a reset was triggered (via `UserInputEvents`), call `IThread.reset`.
5. End the thread if `IThread.is_finished` reports completion.
6. Wait as needed to maintain `IThread.frequency`.
7. End the thread if a close event was set while waiting.
8. Run `IThread.step`.
9. Store the generated step and the last-presented frame for presentation and
   compositing.

### Using `ISession` user-visible-threads

Implement `IThread` and register it from `ISession.init`:

```python
from dataclasses import dataclass

from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.step_result import StepResult

# This is the ID of the game thread.
GAME_THREAD_ID = 1

# GameState is the state of the game thread.
@dataclass
class GameState:
    score: int = 0

    def increment_score(self) -> None:
        self.score += 1

class GameThread(IThread[GameState]):
    def step(self, step_index, events):
        ...

    def reset(self) -> None:
        self.state.score = 0


class ModelThread(IThread[None]):
    def step(self, step_index, events) -> StepResult:
        # Send a message from the model-generation-thread to the game-thread.
        self.invoke_async(
            GAME_THREAD_ID,
            lambda state: state.increment_score(),
        )
        ...


class MySession(ISession):
    def init(self) -> None:
        self.register_model_generation_thread(ModelThread, state=self)
        self.register_thread(
            GAME_THREAD_ID,
            GameThread,
            state=GameState(),
            frequency=60,
        )
        ...
```

Both registration methods construct the requested `IThread` subclass and
register it. `register_model_generation_thread` derives its frequency from
`SessionDesc.frames_per_second_for_step`; `register_thread` accepts an explicit
frequency.
All arguments from `state` and beyond are forwarded unchanged to the constructor
of the used `IThread` subclass.
For example, an `ImGUIThread` registration also has in its constructor `output_layout`, `width`, and `height`:

```python
self.register_thread(
    UI_THREAD_ID,
    MyImGUIThread,
    state=UIState(),
    frequency=self.session_desc.frames_per_second_for_ui,
    output_layout=self.session_desc.output_layout,
    width=self.session_desc.video_width,
    height=self.session_desc.video_height,
)
```

`frequency` is a required non-negative integer giving the maximum number of `step` calls per second.
Zero means unbounded. Each user-visible-thread has its own `step_index`,
which returns to zero after reset. `SessionDesc.frames_per_second_for_step`
supplies this value only for the model-generation-thread; every user-visible-thread
supplies its own value when registered.

`IThread.invoke_async` puts a fire-and-forget `Message`
in the target user-visible-thread's `message_queue`, snapshotting and processing the queue
before the next `step`/`step_ui` of that user-visible-thread.


An operation that raises or returns a value other than `None` fails the user-visible-thread
and triggers shutdown of the entire session. Message queue operations that have not started
when the session stops are discarded.

`IThread.get_model_generation_thread_id` returns the reserved model-generation-thread ID.
`IThread.get_last_presented_frame` returns a shared, read-only `[C, H, W]` tensor.
It returns `None` if the target user-visible-thread has not contributed a frame to the
current generation. Unknown user-visible-thread IDs raise `KeyError`.

### Input events

The io-thread appends window input to one arrival-ordered buffer. Every user-visible-thread
has an independent cursor and receives each retained event once. Periodic garbage
collection removes the prefix every active user-visible-thread has read.

A close event requests session-wide shutdown immediately. A reset event advances
the session generation. Every user-visible-thread resets before its next step, and any result
that was still being produced for the previous generation is not presented.

### UI user-visible-threads and compositing

`UIThread` implements `step` for the user. A subclass implements `step_ui`, which
returns one frame which is ready to be presented.

`flashdreams.runtime_v2.imgui_thread.ImGUIThread` uses
`SlangPyImGUIRenderer`, so an application subclass only supplies the render
dimensions and implements `draw_ui` and any application-state reset.
The renderer owns the ImGui context, external-memory buffer, and synchronization
details on the UI user-visible-thread. `ImGUIThread.draw_frame` places a normalized
`[C, H, W]` video frame inside the active ImGui layout.

Each user-visible-thread publishes its `latest_step` independently. At every io-thread tick, the
runtime snapshots the latest current-generation result from each user-visible-thread, selects
its latest frame, and applies its `PresentationMode`:

- `showPresentation` updates the user-visible-thread's last-presented frame and composites it
  into the client backbuffer.
- `hidePresentation` updates the last-presented frame without affecting the
  client backbuffer.
- `disablePresentation` skips presentation and updating the last-presented frame.

Visible frames composite by ascending thread ID: ID `0` is the bottom layer,
then ID `1`, and so on. RGB layers are opaque; RGBA layers use their alpha
channel. Compositing follows `frames_per_second_for_ui` even while the
model-generation-thread has no new frame.

For sessions with additional user-visible-threads, the compositor eagerly emits
one frame in the session's declared layout. For sessions with only a
model-generation-thread, all processed steps are forwarded deterministically to
the presentation buffer.
This is a lossless path for MP4 output and benchmarking.
`WhenFull.BLOCK` applies back-pressure while presenting it, and
`WhenFull.DROP_OLDEST` replaces its oldest pending composite. Neither policy
paces or drops an individual user-visible-thread's rendering. The output sink remains a
synchronous consumer and needs no capacity API.
