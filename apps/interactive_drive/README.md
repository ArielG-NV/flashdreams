# FlashDreams Interactive Drive application

This workspace package provides the model-neutral Interactive Drive
application and runner contracts. Concrete integrations own scene loading,
simulation, and model rendering, then register a `create_app` factory through
the `flashdreams.applications` entry-point group.

OmniDreams provides the initial runner:

```bash
uv run --package flashdreams-omnidreams flashdreams-run interactive-drive
```

The shared host supplies local-window, null, MP4, and WebRTC output backends.
