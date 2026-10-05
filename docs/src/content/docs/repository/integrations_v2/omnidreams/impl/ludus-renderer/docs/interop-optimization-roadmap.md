---
title: 'Rendering Pipeline Optimization Roadmap'
---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--rendering-pipeline-optimization-roadmap"></a>

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--integration-links"></a>

## Integration links

- **Applications:** [omnidreams applications](../../../../../../models/omnidreams.md#developer-details)
- **Configuration:** [omnidreams configuration](../../../../../../models/omnidreams.md#developer-details)

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--benchmark-baseline-march-2025"></a>

## Benchmark Baseline (March 2025)

Measured on H100 80GB HBM3, 1280x720, CUDA software rasterizer (CudaRaster).
These are historical measurements, not a current performance guarantee; rerun the
benchmark on the current renderer before using them for capacity planning.

| Batch | Render (ms) | Per-query (ms) | Queries/s |
| --- | --- | --- | --- |
| 1 | 6.10 | 6.10 | 164 |
| 2 | 7.07 | 3.53 | 283 |
| 4 | 8.92 | 2.23 | 449 |
| 8 | 12.52 | 1.56 | 639 |
| 16 | 19.93 | 1.25 | 803 |
| 32 | 34.35 | 1.07 | 932 |
| 64 | 67.22 | 1.05 | 952 |
| 128 | 135.30 | 1.06 | 946 |

**Key findings:**
- Per-query cost plateaus at **~1.05ms** for batch >= 32 -- GPU-bound on rasterization
- Fixed overhead per batch call: **~5ms** (kernel launch + state setup)
- GPU->CPU transfer: **42ms for 32 frames (118 MB)** -- larger than the render itself
- Throughput saturates at **~950 queries/s** regardless of batch size

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--where-time-is-spent"></a>

### Where time is spent

```text

Full pipeline for 32 queries at 1280x720 (34.35ms total):

  Fixed overhead (~5ms):
    +-- CUDA kernel launch overhead              ~2ms
    +-- Buffer state setup                       ~2ms
    +-- Synchronization                          ~1ms

  Per-query rasterization cost (~1.05ms x 32 = ~29ms):
    +-- CudaRaster dispatch (polylines)
    +-- CudaRaster dispatch (polygons)
    +-- CudaRaster dispatch (obstacles)
    +-- MSAA resolve (if enabled)

  GPU->CPU transfer (42ms, measured separately):
    +-- cudaMemcpy device->host (118 MB RGBA8)

```

Rasterization dispatches dominate the per-query cost.

---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--phase-2-gpu-cpu-transfer-optimization-high-priority"></a>

## Phase 2: GPU->CPU Transfer Optimization -- HIGH PRIORITY

In the March 2025 baseline, the GPU->CPU transfer (42ms) exceeded the render time
(34ms) for batch=32. Rebenchmark the current RGB output path before treating it
as the largest time sink.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--2a-pinned-page-locked-host-memory"></a>

### 2a: Pinned (page-locked) host memory

Allocate a host staging tensor in pinned memory and copy into it asynchronously so
the device-to-host transfer can use DMA:

```python

output = torch.empty(gpu_images.shape, dtype=gpu_images.dtype, pin_memory=True)
output.copy_(gpu_images, non_blocking=True)

```

Historical estimate: 2-3x faster for large transfers; verify on the target system.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--2b-async-staging-with-double-buffering"></a>

### 2b: Async staging with double buffering

Overlap GPU->CPU transfer with the next batch's render:

```text

Batch N:   Render(N) --> Copy-to-staging(N) --> DMA-to-host(N)
Batch N+1:               Render(N+1) ----------> Copy-to-staging(N+1)
                              (overlapped)

```

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--2c-skip-transfer-entirely-gpu-resident-pipeline-available"></a>

### 2c: Skip transfer entirely (GPU-resident pipeline) -- AVAILABLE

If the consumer is a GPU model (training or inference), keep the rendered images
as GPU tensors. `render_sequence_gpu()` and `LudusCudaTimestampedContext.render()`
already return CUDA tensors; avoid the `gpu_to_numpy()` convenience conversion to
eliminate the historical 42ms transfer completely.

---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--phase-3-rasterization-optimization-high-priority"></a>

## Phase 3: Rasterization Optimization -- HIGH PRIORITY

The 1.05ms per-query floor at high batch sizes means the CudaRaster kernel
dispatches are the GPU bottleneck.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--3a-profile-with-nsys"></a>

### 3a: Profile with nsys

```bash

nsys profile --trace=cuda \
  uv run python examples/benchmark_renderer.py --scene ... --iters 10

```

Identify which rasterization pass (polyline, polygon, obstacle) dominates and
whether there are idle gaps between kernel launches.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--3b-reduce-overdraw-early-exit"></a>

### 3b: Reduce overdraw / early-exit

Scene upload now computes the actual maximum varray count per timestamp from each
scene's prefix sums, replacing the old fixed `MAX_VARRAYS_PER_POOL`-style bound.
The renderer still dispatches that scene-wide maximum for every pool and query, so
some invocations early-exit when a particular pool/timestamp has less data. A
remaining optimization would use per-pool or per-query exact dispatch counts.

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--3c-frustum-culling"></a>

### 3c: Frustum culling

The renderer already computes per-element AABBs for polyline and polygon pools and
applies radius-based spatial culling (plus sphere/radius culling for cubes). True
camera-frustum culling is still a possible follow-up for limited-FOV, non-BEV
cameras.

---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--phase-4-scene-upload-optimization-low-priority"></a>

## Phase 4: Scene Upload Optimization -- LOW PRIORITY

In the March 2025 baseline, scene upload was a one-time cost (~0.5s for small
scenes, ~22s for large scenes), not a per-frame bottleneck. The current renderer
also has GPU parquet decoding, tar prefetching, and L1/L2/L3 scene caches, so
remeasure uncached and cached loading separately.

Remaining potential improvement:
- Streaming upload (start rendering before the full scene is loaded)

---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--phase-5-multi-process-multi-gpu-future"></a>

## Phase 5: Multi-Process / Multi-GPU -- FUTURE

For training at scale, separate scene loading and rendering across processes or
GPUs. Relevant only after per-frame bottlenecks are addressed.

---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--historical-priority-summary"></a>

## Historical Priority Summary

```text

Impact vs Effort:

HIGH IMPACT, MODERATE EFFORT:
  Phase 2a: Pinned memory transfers        -> saves ~20ms per batch
  Phase 2c: GPU-resident pipeline          -> available; historically saved ~42ms
  Phase 3a: nsys profiling                 -> identifies rasterization bottleneck

HIGH IMPACT, HIGH EFFORT:
  Phase 3b-c: Remaining raster optimization -> targets 1.05ms/query historical floor
  Phase 2b: Async staging                  -> overlaps transfer with render

LOW IMPACT:
  Phase 4: Scene upload                    -> one-time cost, not per-frame

```

---

<a id="integrationsv2-omnidreams-impl-ludus-renderer-docs-interop-optimization-roadmap--measuring-progress"></a>

## Measuring Progress

| Metric | How to measure | Current baseline |
| --- | --- | --- |
| Per-query render cost | `benchmark_renderer.py --iters 10` | 1.05ms @ batch=32 |
| Fixed overhead per batch | batch=1 minus (per-query x 1) | ~5ms |
| GPU-&gt;CPU transfer | `render_all_frames(...)` timing key `cpu_transfer` | 42ms for 32x1280x720 (historical) |
| GPU utilization | `nsys` trace -- idle gaps between CUDA kernels | Not yet measured |
| End-to-end FPS | `render_hdmap_scene.py` timing output | ~950 queries/s |
