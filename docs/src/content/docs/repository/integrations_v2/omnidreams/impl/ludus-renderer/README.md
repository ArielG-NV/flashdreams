---
title: 'Ludus Renderer'
---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--ludus-renderer"></a>

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--integration-links"></a>

## Integration links

- **Applications:** [omnidreams applications](../../../../../models/omnidreams.md#developer-details)
- **Configuration:** [omnidreams configuration](../../../../../models/omnidreams.md#developer-details)

GPU-native F-theta renderer and PhysX-first object graph for autonomous
vehicle simulation. Rendering is CUDA-only and is built on the HPG 2011
CudaRaster triangle pipeline.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--features"></a>

## Features

- **F-theta Camera Model**: Native support for fisheye lens distortion using polynomial projection
- **One CUDA rendering path**: no graphics API interop or duplicate backend
- **PhysX object graph**: generic rigid bodies and invisible map barriers;
  applications own semantic vehicle design and driving policy
- **Mutable simulation buffers**: actor add/update/remove operations preserve one
  native scene and one fixed-capacity state table
- **Timestamped Rendering**: Efficient temporal queries for simulation playback
- **Adaptive Tessellation**: Automatic subdivision based on distortion error
- **MSAA**: configurable CUDA supersampling
- **Mirror Augmentation**: Extend scenes by tiling reflected copies for longer driving sequences
- **GPU Spatial Culling**: Per-element AABB/sphere culling for large scenes

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--primitives"></a>

## Primitives

- **Polylines**: Thick line strips with configurable width and round caps
- **Polygons**: Filled polygons with pre-triangulation
- **Cubes**: Oriented bounding boxes with 9-DOF transform

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--requirements"></a>

## Requirements

**Always:**

- NVIDIA GPU (Turing or later)
- An NVIDIA driver compatible with the installed PyTorch CUDA build (R580 or
  newer for the workspace's default CUDA 13 profile)
- A CUDA Toolkit matching that PyTorch build (13.x by default)
- Python 3.10-3.13
- A C++17 compiler (Visual Studio 2022 on Windows)

MP4 output also requires ffmpeg. Video overlay examples require the `dev`
extra, which installs OpenCV and a bundled ffmpeg fallback.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--installation"></a>

## Installation

```bash

cd integrations_v2/omnidreams/impl/ludus-renderer
uv sync --package ludus-renderer --extra dev

```

Dependencies installed:

- PyTorch 2.11+
- NumPy, Pandas, SciPy
- PyArrow (for parquet scene files)
- Pillow (for image handling)
- OpenCV and imageio-ffmpeg from the `dev` extra (for video overlays and the
  bundled ffmpeg fallback)
- CMake 4+ for the one-time native module build
- NVIDIA PhysX 5.9.0, fetched from the official repository at a pinned commit
  and verified by SHA-256

Prebuild PhysX during environment setup so interactive startup never pays the
one-time compile cost:

```bash

uv run --package ludus-renderer python -c "from ludus_renderer import prepare_physx; prepare_physx()"

```

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--usage"></a>

## Usage

The CUDA renderer is the only public rendering backend:

```python

from ludus_renderer import LudusCudaTimestampedContext
ctx = LudusCudaTimestampedContext(device="cuda")

```

`PhysXWorld` consumes `PhysicsObjectGraph` and creates `PxScene`, rigid actors,
materials, and shapes directly, without an intermediate scene-format runtime.
The first PhysX use in each process asks CMake to configure and build the pinned
CPU module. CMake reuses the platform build cache and only rebuilds targets that
are out of date; later PhysX worlds in the same process reuse the already-loaded
module without invoking CMake again. Set `LUDUS_PHYSX_CACHE` to place this cache
elsewhere.

`PhysicsObjectGraph.upsert_object`, `remove_object`, `upsert_barrier`, and
`remove_barrier` edit graph topology. `PhysXWorld.synchronize` applies only those
differences to the live scene. Its `state_buffer` and `active_buffer` views keep
the same addresses while body slots are recycled. `MutableObjectSceneBuffer`
owns the CUDA renderer's update-versus-reallocate policy, so interactive clients
do not manage packed-scene replacement themselves.

`build_hdmap_object_pool` turns simulated graph tracks into the same oriented
boxes used by regular-camera and BEV model inputs. `PhysicsObjectGraph.copy_for_physx`
creates a bounded active topology while the complete graph remains available to
render model-input and BEV boxes.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--examples"></a>

## Examples

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--hdmap-scene-renderer"></a>

### HDMap Scene Renderer

Render clipgt HDMap scenes with road geometry, lane lines, obstacles, and traffic elements:

```bash

# Render a single frame

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene --frame 12

# Render bird's eye view

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene --frame 12 --bev

# Render full sequence to PNG frames

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene --sequence

# Render all cameras at 30fps as MP4

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene --sequence --all-cameras --fps 30 --output-format mp4

# Render specific camera with JPEG output

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene --sequence --camera camera:front:wide:120fov --output-format jpg

# Enable 4x antialiasing

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene --sequence --msaa 4

```

**Key options:**
- `--msaa N`: MSAA sample count (`0` = disabled, `4` = 4x antialiasing)
- `--camera NAME`: Render from a specific scene camera (use `--list-cameras` to see available)
- `--all-cameras`: Render from all available cameras in the scene
- `--fps N`: Output frame rate in Hz (default: 10)
- `--output-format`: `png` (default), `jpg` (nvJPEG hardware encode), or `mp4` (H264 via ffmpeg libx264)
- `--batch-size N`: Number of frames to render per GPU batch (default: all frames at once)
- `--quality N`: JPEG quality 1-100 (default: 90)
- `--bitrate N`: MP4 bitrate in bps (default: 10Mbps)

Scene elements rendered:
- Road boundaries, lane lines (solid/dashed/dotted, white/yellow)
- Crosswalks, road markings, wait lines
- Traffic lights, traffic signs, poles
- Dynamic obstacles (vehicles, pedestrians)
- Ego trajectory and BEV ego vehicle

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--video-overlay"></a>

### Video Overlay

Composite rendered HD map elements on top of an input video (50:50 blend). Supports all output formats:

```bash

# Overlay as JPEG frames (GPU-accelerated via nvjpeg)

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene \
    --overlay-video /path/to/input-video.mp4 \
    --output-format jpg

# Overlay as MP4 video

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene \
    --overlay-video /path/to/input-video.mp4 \
    --output-format mp4

# Overlay as PNG frames

uv run python examples/render_hdmap_scene.py --scene /path/to/clipgt-scene \
    --overlay-video /path/to/input-video.mp4 \
    --output-format png

```

Blending is performed on GPU using PyTorch integer arithmetic. For JPEG output, encoding uses nvjpeg hardware acceleration. Frame count is capped to `min(rendered frames, video frames)`.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--mirror-augmentation"></a>

### Mirror Augmentation

Extend a scene by mirror-stitching it N times at load time. Two canonical tiles (original + single reflection) are placed alternately via rigid body transforms, producing an `[original]-[mirror]-[original]-[mirror]-...` pattern without rotational drift on curved roads.

```python

from ludus_renderer import load_clipgt_scene, mirror_augment_scene

scene = load_clipgt_scene("/path/to/clipgt-scene", device="cuda")
extended = mirror_augment_scene(scene, n_mirrors=10, lookahead_m=50.0)

```

- `n_mirrors`: number of augmentation iterations (total segments = n_mirrors + 1)
- `lookahead_m`: distance (metres) beyond the ego endpoint to place the first mirror plane

GPU-side spatial culling avoids rasterizing distant elements. Culling is enabled
by default (1.5x `depth_max`) and can be adjusted via:

```python

ctx.set_cull_radius(scale=1.5)  # 0 disables culling

```

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--benchmarking"></a>

### Benchmarking

```bash

# Single scene benchmark

uv run python examples/benchmark_renderer.py --scene /path/to/clipgt-scene --iters 10

# Multi-camera benchmark (8 cameras per timestamp)

uv run python examples/benchmark_renderer.py --scene /path/to/clipgt-scene --multicam

```

<a id="integrationsv2-omnidreams-impl-ludus-renderer-readme--contributing"></a>

## Contributing

Contributions are welcome, thank you. This project only accepts contributions under the
Apache License, Version 2.0. All contributions must be signed off in accordance with the
[Developer Certificate of Origin (DCO)](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/omnidreams/impl/ludus-renderer/contributing).
