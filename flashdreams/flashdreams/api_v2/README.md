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
window until the model-generation-thread reports `is_finished`, the window
reports a close, or the caller-requested step count is reached. A caller holding
an application uses
`flashdreams.runtime_v2.application_runner.ApplicationRunner` to get there,
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
  carry into another run. During `init`, it registers exactly one
  model-generation-thread and any additional user-visible-threads. The
  model-generation-thread says when the run is over through `is_finished`; the
  default is to never finish.
- `InputSource` and `OutputSink` belong to the runtime. The runner reads from the
  source, passes `UserInputEvents` to each registered `IThread`, and routes each
  `StepResult` to presentation. Neither a session nor its threads own the source
  or sink.
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
- `ISession` owns the session lifecycle, its user-visible-threads, and one
  `PresentationCordinator` for the run. Call
  `ISession.get_presentation_cordinator` to access it.
- The registered model-generation-thread executes under its automatically
  reserved, process-unique thread ID.
- `ISession` defines `init` to register threads and `close` to clean-up any auxillary resources initialized in `init`.

Program Threads:
  - `main-program-thread`:

    - Runs the session through `run_session`, called through (generally) a provided
      `create_app` method.
    - Opens and owns the client window, collects input, and ticks presentation
      at `SessionDesc.frames_per_second_for_ui`.
    - Launches all user-visible-threads after the client window is open.
    - Rejoins the threads launched with the rest of the demo when signals are appropriately received (ctrl+c, model-generation-thread finishes, etc...).
    - Merges presentation-ready frames from each user-visible-thread into a
      single frame through the session's `PresentationCordinator`. The
      coordinator drains into the client window as often as the window permits,
      subject to back-pressure.
      - The `run_session` `when_full` argument controls what happens when a
        frame is generated while the coordinator is full.
    - `ISession.init` must call
      `set_layer_order_via_thread_id(thread_id_list)` after registration. Index
      zero is the bottom layer and the final index is the top layer.
    - When compositing, queue up frames for lossless presentation if only model-generation-thread is present. This is primarily for testing purposes where visual consistency is important.

- `user-visible-threads`:

  - `model-generation-thread`:

    - Is an `IThread` explicitly registered by `ISession.init` through
      `ISession.register_main_generation_thread`.
    - Has its automatically assigned ID available as
      `ISession.main_generation_thread_id` and as the registration return value.
    - Ticks `step` at rate of
      `SessionDesc.frames_per_second_for_step`.

  - All user-visible-threads except for the model-generation-thread:

    - Register with
      `ISession.register_thread(thread_type, state=..., frequency=..., ...)`
      during `ISession.init`. Registering elsewhere raises an exception.
      Registration automatically reserves and returns a process-unique thread ID.
      Arguments depend on the type of thread being registered; all arguments
      from `state` onward are forwarded to the `IThread` implementation's
      `__init__` method.
    - Ticks `step` at rate of `IThread.frequency`.
    - Register via implementing `IThread` or a subclass such as `UIThread` or `ImGUIThread`. Refer to `integrations_v2\imgui_demo\imgui_demo\app.py` for an example.

  - All user-visible-threads:
    - Communicate only through
      `flashdreams.invoke_async(thread_id, lambda state: ...)`, where `state` is the target thread's
      `IThread.state` (typed by `IThread.StateT`). This method adds a message to
      a thread's `message_queue`. The process-wide lookup holds a weak reference;
      the owning `ISession` controls the target thread's lifetime.
    - Message queue processes via: 1. Snapshot the queue, 2. Processing the snapshot, 3. Clearing the processed messages.
    - After the source thread is registered, obtain its stable, read-only
      `PresentedFrame` handle during `ISession.init` through
      `ISession.get_presentation_cordinator().get_last_presented_frame(thread_id)`.
      Pass the handle to another user-visible-thread and call `get` to paint the
      source thread's frame with `ImGUIThread.draw_frame`.
    - `flashdreams.invoke_async` can queue startup messages in `ISession.init`
      for user-visible-threads to execute before their first `step` call.

### Lifecycle

`main-program-thread`:

1. Call `ISession.init`, which registers the model-generation-thread and any
   additional user-visible-threads.
2. Open the client window and read its input into `event_buffer`.
3. Start the user-visible-threads.
4. At each `SessionDesc.frames_per_second_for_ui` tick:

   1. Read client-window input into `event_buffer`.
   2. Collect `event_buffer` garbage.
   3. Compose the next presentable frame from all user-visible-threads.
   4. Add the frame to `PresentationCordinator` using the `when_full` policy.
   5. Drain `PresentationCordinator` into the client window.

5. Stop and join the user-visible-threads, clear `event_buffer`, close the
   client window, and call `ISession.close`.

Each `user-visible-thread` repeats the following loop:

1. Snapshot `message_queue` and process the messages in that snapshot.
2. Read new user events and the current generation from `event_buffer`.
3. If the events contain a close event, request session-wide shutdown and leave
   the loop.
4. If the generation changed because of a reset event:

   1. Call `IThread.reset`.
   2. Discard pending results. `PresentationCordinator` clears the value exposed
      by every `PresentedFrame` handle for the previous generation.
   3. Reset the thread's `step_index` to zero.

5. Leave the loop if `IThread.is_finished` reports completion.
  - If completed thread is model-generation-thread, the session is finished and will propegate a session-wide shutdown event to all other user-visible-threads.
6. Wait as needed to maintain `IThread.frequency`, leaving the loop if
   session-wide shutdown is requested while waiting.
7. Call `IThread.step` with the current `step_index` and event batch.
8. If shutdown was requested while `step` ran, discard its result and leave the
   loop. Otherwise, publish the result under the current generation for the
   main-program-thread to present or composite, then increment `step_index`.

When the loop ends, whether because of close input, completion, failure, a
model-generation step limit, or session-wide shutdown, the user-visible-thread
closes its thread-owned resources, stops accepting messages, discards messages
that have not started, and unregisters from `event_buffer`. After every
user-visible-thread has stopped, the main-program-thread calls `ISession.close`
for session-owned resources.

### Using `ISession` user-visible-threads

Implement `IThread` and register it from `ISession.init`:

```python
from dataclasses import dataclass

from flashdreams import invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.step_result import StepResult

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


@dataclass(frozen=True)
class ModelState:
    game_thread_id: int


class ModelThread(IThread[ModelState]):
    def step(self, step_index, events) -> StepResult:
        # Send a message from the model-generation-thread to the game-thread.
        invoke_async(
            self.state.game_thread_id,
            lambda state: state.increment_score(),
        )
        ...


class MySession(ISession):
    def init(self) -> None:
        game_thread_id = self.register_thread(
            GameThread,
            state=GameState(),
            frequency=60,
        )
        main_generation_thread_id = self.register_main_generation_thread(
            ModelThread,
            state=ModelState(game_thread_id=game_thread_id),
        )
        self.set_layer_order_via_thread_id(
            [main_generation_thread_id, game_thread_id]
        )
        ...
```

Both registration methods construct the requested `IThread` subclass, reserve
an ID with `flashdreams.reserve_thread_id()`, register it, and return that ID.
Callers cannot supply a thread ID to either registration method.
`register_main_generation_thread` derives its frequency from
`SessionDesc.frames_per_second_for_step`; `register_thread` accepts an explicit
frequency.
After registering every thread, `ISession.init` must call
`set_layer_order_via_thread_id` with every returned ID exactly once. The list is
the explicit bottom-to-top compositing order and is unrelated to numeric ID
order.
All arguments from `state` and beyond are forwarded unchanged to the constructor
of the used `IThread` subclass.
For example, an `ImGUIThread` registration also has in its constructor `output_layout`, `width`, and `height`:

```python
ui_thread_id = self.register_thread(
    MyImGUIThread,
    state=UIState(),
    frequency=self.session_desc.frames_per_second_for_ui,
    output_layout=self.session_desc.output_layout,
    width=self.session_desc.video_width,
    height=self.session_desc.video_height,
)
self.set_layer_order_via_thread_id(
    [self.main_generation_thread_id, ui_thread_id]
)
```

`frequency` is a required non-negative integer giving the maximum number of `step` calls per second.
Zero means unbounded. Each user-visible-thread has its own `step_index`,
which returns to zero after reset. `SessionDesc.frames_per_second_for_step`
supplies this value only for the model-generation-thread; every user-visible-thread
supplies its own value when registered.

`flashdreams.invoke_async` puts a fire-and-forget `Message`
in the target user-visible-thread's `message_queue`, snapshotting and processing the queue
before the next `step`/`step_ui` of that user-visible-thread.


An operation that raises or returns a value other than `None` fails the user-visible-thread
and triggers shutdown of the entire session. Message queue operations that have not started
when the session stops are discarded.

`ISession.main_generation_thread_id` exposes the automatically assigned
model-generation-thread ID. Thread registration also registers that thread with
the session's presentation coordinator. After registration,
`ISession.get_presentation_cordinator().get_last_presented_frame(thread_id)`
returns a stable, read-only `PresentedFrame` handle. Its `get` method returns
a `[C, H, W]` tensor, or `None` if the target user-visible-thread has
not contributed a frame to the current generation. Unknown IDs raise
`KeyError`. The handle remains the same object across presentation updates. During an update, its underlying value changes. During a `reset`, its value is cleared to `None`.

For example, a session can pass the model-generation-thread's handle into an
ImGui-thread's state while registering that thread:

```python
main_generation_thread_id = self.register_main_generation_thread(
    ModelThread,
    state=model_state,
)
model_frame = self.get_presentation_cordinator().get_last_presented_frame(
    main_generation_thread_id
)
ui_thread_id = self.register_thread(
    MyImGUIThread,
    state=UIState(model_frame=model_frame),
    frequency=self.session_desc.frames_per_second_for_ui,
    output_layout=self.session_desc.output_layout,
    width=self.session_desc.video_width,
    height=self.session_desc.video_height,
)
self.set_layer_order_via_thread_id(
    [main_generation_thread_id, ui_thread_id]
)
```

### Input events

The main-program-thread appends window input to one arrival-ordered buffer. Every user-visible-thread
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

Each user-visible-thread publishes its `latest_step` independently. At every main-program-thread tick, the
runtime snapshots the latest current-generation result from each user-visible-thread, selects
its latest frame, and applies its `PresentationMode`:

- `SHOW_PRESENTATION` updates the user-visible-thread's `PresentedFrame` value
  and uses it to compose a presentation into the client backbuffer.
- `HIDE_PRESENTATION` updates `PresentedFrame` value without affecting our final composed frame for presentation.
- `DISABLE_PRESENTATION` skips frame extraction, `PresentedFrame` updates, and
  final frame composition for presentation.

Visible frames composite in the exact order supplied to
`ISession.set_layer_order_via_thread_id`: index zero is the bottom layer and the
final index is the top layer. Numeric thread-ID order has no effect. RGB layers
are opaque; RGBA layers use their alpha channel. Compositing follows
`frames_per_second_for_ui` even while the model-generation-thread has no new
frame.

For sessions with additional user-visible-threads, the compositor eagerly emits
one frame in the session's declared layout. For sessions with only a
model-generation-thread, all processed steps are forwarded deterministically to
the presentation buffer.
This is a lossless path for MP4 output and benchmarking.
`WhenFull.BLOCK` applies back-pressure while presenting it, and
`WhenFull.DROP_OLDEST` replaces its oldest pending composite. Neither policy
paces or drops an individual user-visible-thread's rendering. The output sink remains a
synchronous consumer and needs no capacity API.
