GPU: NVIDIA GB300

| Checkpoint | Backend | Latent shape C×T×H×W | Latents/chunk | Denoise steps/chunk | Startup (s) | Generated FPS | Median request (s) | p90 request (s) | Video |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cosmos3-Nano-Sim-Bimanual | FlashDreams | 48 × 25 × 22 × 40 | 2 | 4 | 19.39 | 17.03 | 5.59 | 5.80 | [Watch](results/three-seconds/bimanual/flashdreams/output.mp4) |
| Cosmos3-Nano-Sim-Bimanual | vLLM-Omni | 48 × 25 × 22 × 40 | 2 | 4 | 615.69 | 10.06 | 9.48 | 10.19 | [Watch](results/three-seconds/bimanual/vllm/output.mp4) |
| Cosmos3-Nano-Sim-Auto | FlashDreams | 48 × 25 × 30 × 52 | 3 | 4 | 36.74 | 10.92 | 8.80 | 8.89 | [Watch](results/three-seconds/auto/flashdreams/output.mp4) |
| Cosmos3-Nano-Sim-Auto | vLLM-Omni | 48 × 25 × 30 × 52 | 3 | 4 | 525.22 | 8.55 | 11.37 | 12.25 | [Watch](results/three-seconds/auto/vllm/output.mp4) |

Startup: saved benchmark.json startup_s, covering model/backend initialization before the first warmup request; excludes process launch, input preparation and request warmup. Cache state can affect startup time.

Videos: 97 frames at 30 FPS, including the initial frame; last measured request, seed 0. MP4 encoding is excluded from timings. Auto/AV uses one RGB camera without LiDAR. Runs use eager denoising with CUDA graphs off; these are not the notebooks' optimized serving settings. Quality equivalence is unverified. Measurements on a shared GPU can vary.
