# FlashDreams interactive-drive application

The shared interactive-drive package owns the transport-neutral application,
session, canonical driver controls, and presentation HUD. Model integrations
provide a small factory with their adapter, conditioning provider, scenario
type, and runtime defaults.

OmniDreams is the only concrete integration currently registered. Install and
launch it with:

```bash
uv run --package flashdreams-omnidreams flashdreams-run interactive-drive
```

The application works with every FlashDreams application output backend. Use
`--no-ui` to omit the presentation-only driving HUD.

## Command-line options

The complete command-line shape is:

```bash
uv run --package flashdreams-omnidreams flashdreams-run interactive-drive \
  [HOST_OPTIONS] [APPLICATION_OPTIONS]
```

Options may appear in any order after `interactive-drive`. The defaults below
are the values supplied by the currently registered OmniDreams integration.

### Host and output options

These options are handled by the FlashDreams application host before the
remaining arguments are passed to interactive-drive.

| Option | Default | Description |
|---|---|---|
| `--output {local-window,null,mp4,webrtc}` | `local-window` | Select the presentation/output backend. |
| `--output-path PATH` | `outputs/interactive-drive.mp4` | MP4 destination. This option is available with non-WebRTC outputs and is used only by `mp4`. |
| `--output-fps FLOAT` | application FPS | Override the local-window or MP4 presentation frame rate. This option is available with non-WebRTC outputs. |
| `--host HOST` | `127.0.0.1` | WebRTC bind host. Accepted only with `--output webrtc`. |
| `--port PORT` | `8080` | WebRTC bind port, from 1 through 65535. Accepted only with `--output webrtc`. |

### Application options

| Option | OmniDreams default | Description |
|---|---|---|
| `-h`, `--help` | — | Show the interactive-drive application help and exit. |
| `--preset-id ID` | `omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae` | Select the integration-owned inference preset. |
| `--device DEVICE` | `cuda` | Device used to construct the model runtime, such as `cuda`, `cuda:1`, or `cpu`. |
| `--seed INTEGER` | `42` | Set the inference seed. |
| `--scene-path PATH` | unset | Use a specific local scene asset. |
| `--scene-dir PATH` | unset | Search a local directory for scene assets. |
| `--scene-uuid UUID` | `0d404ff7-2b66-498c-b047-1ed8cded60d4` | Select a remotely discoverable scene. |
| `--scene-variant VARIANT` | `default` | Select a visual variant of the scene, such as `default`, `rain`, or `snow`. |
| `--camera-name NAME` | `camera_front_wide_120fov` | Select the scene camera rendered by the conditioning provider. |
| `--prompt TEXT` | unset | Override the model prompt for the selected scene. |
| `--total-blocks INTEGER` | `2147483647` | Stop after this many model chunks. Must be greater than zero. |
| `--pixel-height INTEGER` | `704` | Set the output frame height. Must be greater than zero. |
| `--pixel-width INTEGER` | `1280` | Set the output frame width. Must be greater than zero. |
| `--fps INTEGER` | `30` | Set the model presentation and control-sampling frame rate. Must be greater than zero. |
| `--move-speed-per-s FLOAT` | `6.0` | Set the scene-provider translation speed per second. Must be greater than zero. |
| `--rotate-speed-rad-per-s FLOAT` | `0.6108652381980153` | Set the scene-provider rotation speed in radians per second. Must be greater than zero. |
| `--scene-backend {cuda,vulkan}` | `cuda` | Select the scene rendering backend. |
| `--ludus-backend {cuda,vulkan}` | `cuda` | OmniDreams alias for `--scene-backend`. |
| `--ui`, `--no-ui` | `--ui` | Enable or disable the presentation-only driving HUD. |

For example, to serve the demo over WebRTC with the driving HUD:

```bash
uv run --package flashdreams-omnidreams flashdreams-run interactive-drive \
  --output webrtc --host 0.0.0.0 --port 8080 --no-ui
```
