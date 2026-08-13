# FlashDreams text-to-video application base

This package implements the reusable, transport-neutral T2V application and
session lifecycle. It does not select or import a model integration.

Each model package owns its concrete application factory and registers its full
slug:

- `cosmos_predict2.t2v.app:createApp` → `t2v-cosmos-predict2`
- `causal_forcing.t2v.app:createApp` → `t2v-causal-forcing`
- `self_forcing.t2v.app:createApp` → `t2v-self-forcing`

For example:

```console
uv run flashdreams run t2v-cosmos-predict2 --prompt "A robot welding."
```

The concrete integration provides the pipeline and geometry defaults.
`--prompt` is required; `--total-blocks`, `--pixel-height`,
`--pixel-width`, `--device`, and `--compile` override its defaults.

The reusable session sends each generated tensor directly to the host-provided
`OutputSink`. It does not import WebRTC, open files, or assume any particular
presentation backend.
