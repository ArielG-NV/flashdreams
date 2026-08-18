# FlashDreams T2V applications

The shared T2V package provides the application/session protocol; each model
integration owns its small `t2v/app.py` factory. `--prompt` is optional for
interactive local-window and WebRTC runs, and remains available to seed the
first generation. Non-interactive outputs such as MP4 still need a prompt.

## Server-rendered prompt UI

Local-window and WebRTC runs now render a Dear ImGui prompt editor on the
server and alpha-composite it into the same frames sent to every output
backend. When `--prompt` is omitted, the presenter opens immediately on a black
idle frame and generation waits for a non-empty prompt submitted with
**Generate**. When supplied, the command-line prompt seeds the first generation.
While a generation is running, edit the prompt and select **Generate**; the
semantic request is retained across the reusable session boundary and the next
model cache is initialized with the submitted text on the model thread. ImGui
input and frame presentation continue on the presentation thread while the
model is inside `generate`.

For a targeted workspace environment, select the integration distribution in
the `uv run` command. This syncs the integration, `flashdreams-t2v`, and the
local-window/serving dependencies without installing unrelated integrations:

- `t2v-causal-forcing` → `--package flashdreams-causal-forcing`
- `t2v-cosmos-predict2` → `--package flashdreams-cosmos-predict2`
- `t2v-self-forcing` → `--package flashdreams-self-forcing`

Native SlangPy window (default, presented at 60 FPS):

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing
```

Use the global `--presentation-fps` option to choose a different UI and
backend cadence. Model frames retain their configured source FPS and are held
and recomposited between updates:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --presentation-fps 30
```

The same demo can be launched from Python with the application runner used by
`flashdreams-run t2v-causal-forcing`:

```python
from flashdreams.demo import run_application

run_application(
    "t2v-causal-forcing",
    [],
)
```

WebRTC browser backend:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --output webrtc --host 0.0.0.0 --port 8080
```

Then open `http://localhost:8080/request_session`.

MP4 artifact:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --output mp4 --output-path artifacts/output.mp4 \
  --prompt "A robot walking through a forest."
```

Available slugs are `t2v-cosmos-predict2`, `t2v-causal-forcing`, and
`t2v-self-forcing`. All backends receive the same transport-neutral
`InputHandler` and `OutputSink` API. Input handlers publish named,
time-windowed `CanonicalInputWindow` values matching each application's
`CanonicalInputSchema`.
