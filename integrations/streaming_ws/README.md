# `streaming_ws`

Small demo: **control → batched WebP frames**, with optional **Viser** browser path (no WebSocket client).

## Install

From the repo root:

```bash
uv pip install --system --break-system-packages --no-build-isolation \
  -e "integrations/streaming_ws[dev]"

# Optional: OpenCV window on the WS client
uv pip install --system --break-system-packages --no-build-isolation \
  -e "integrations/streaming_ws[streaming_viewer]"

# Optional: Viser-only “client” in the browser
uv pip install --system --break-system-packages --no-build-isolation \
  -e "integrations/streaming_ws[streaming_viser]"
```

## WebSocket server + client

Terminal A — listen on all interfaces (default port `8765`):

```bash
python -m streaming_ws server
```

Terminal B — connect and stream (defaults to `ws://127.0.0.1:8765`):

```bash
python -m streaming_ws client
```

Useful flags (see full lists with `--help`):

```bash
python -m streaming_ws server --help
python -m streaming_ws client --help
python -m streaming_ws client --uri ws://<host>:8765 --show-window
```

The server prints `ws://` URLs for LAN hosts when bound to `0.0.0.0`.

## Viser browser client (no WS)

Single process: **camera movement** in the viewer schedules mock batches; frames are shown with **`client.scene.set_background_image`** (not a GUI image panel).

```bash
python -m streaming_ws viser
```

Then open the printed `http://` URL (e.g. `http://127.0.0.1:8082`). Orbit/pan the camera to pull more frames; optional flat depth plane:

```bash
python -m streaming_ws viser --flat-background-depth-m 10.0
```

```bash
python -m streaming_ws viser --help
```
