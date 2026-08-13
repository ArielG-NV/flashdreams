# FlashDreams T2V applications

The shared T2V package provides the application/session protocol; each model
integration owns its small `t2v/app.py` factory. A non-empty `--prompt` is
required.

Native SlangPy window (default):

```console
uv run flashdreams run t2v-causal-forcing \
  --prompt "A robot walking through a forest."
```

WebRTC browser backend:

```console
uv run flashdreams run t2v-causal-forcing \
  --output webrtc --host 0.0.0.0 --port 8080 \
  --prompt "A robot walking through a forest."
```

Then open `http://localhost:8080/request_session`.

MP4 artifact:

```console
uv run flashdreams run t2v-causal-forcing \
  --output mp4 --output-path artifacts/output.mp4 \
  --prompt "A robot walking through a forest."
```

Available slugs are `t2v-cosmos-predict2`, `t2v-causal-forcing`, and
`t2v-self-forcing`. All backends receive the same transport-neutral
`InputSink` and `OutputSink` API.
