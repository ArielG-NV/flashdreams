# Cosmos3: FlashDreams vs. vLLM

From the repository root on a CUDA Linux host with `uv`, FFmpeg/FFprobe and model/dataset access:

1. Set up the environments:

   ```bash
   bash benchmarks/cosmos3/setup.sh flashdreams
   bash benchmarks/cosmos3/setup.sh auto
   bash benchmarks/cosmos3/setup.sh vllm
   ```

2. Run both checkpoints through FlashDreams and vLLM-Omni and save the comparison table and four videos:

   ```bash
   bash benchmarks/cosmos3/run.sh --skip-setup
   ```

   Inputs are prepared automatically. [RESULTS.md](RESULTS.md) contains one row per
   checkpoint/backend, with latent shape, latents per chunk, denoising steps,
   generated FPS, median/p90 request latency and a video link. Each run saves
   `results/three-seconds/<model>/<backend>/output.mp4` and `benchmark.json`.
   The two warmup requests are excluded; the final measured request supplies the
   video. Encoding and video validation happen outside the timed interval.
   Checkpoint loading and first-run kernel compilation can take several minutes;
   progress and failures are saved in the printed per-backend log paths.
   For setup and benchmarking in one command, use `bash benchmarks/cosmos3/run.sh`.

3. Open the table with an inline video player for each row. Start the results website, then open <http://localhost:8765>:

   ```bash
   python3 benchmarks/cosmos3/results-site.py
   ```

   For a remote GPU host, forward the page to your computer with
   `ssh -L 8765:127.0.0.1:8765 <host>` before opening that URL.

This compares `nvidia/Cosmos3-Nano-Sim-Bimanual` and
`nvidia/Cosmos3-Nano-Sim-Auto` (AV), each at the pinned revision below. Auto
currently uses **one RGB camera without LiDAR**. The [upstream Auto notebook](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/simulation/auto/run_sim_auto_with_vllm_omni.ipynb)
uses seven cameras and optionally joint LiDAR; this smaller workload is not a
reproduction of its full request. The [bimanual notebook](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/simulation/bimanual/run_bimanual_with_vllm_omni.ipynb)
also uses different serving optimizations. Here both backends use eager denoising
and disable CUDA graphs, so results describe this matched configuration.

Both backends use these settings within each model pair. Latent dimensions
are channels × time × height × width; history capacity includes the initial frame.

| Setting | Bimanual: both backends | Auto: both backends |
| --- | --- | --- |
| Checkpoint revision | `be4e2bc243f9f4a0b213eeebd1308cfbac8d9d7e` | `c3fa1378e93710242501a6288d7c9e7c9084a580` |
| GPUs | 1 | 1 |
| Batch size / views | 1 / 1 | 1 / 1 |
| Resolution (width × height) | 640 × 352 | 832 × 480 |
| New / total pixel frames | 96 / 97 | 96 / 97 |
| Full request latent shape (C×T×H×W) | 48 × 25 × 22 × 40 | 48 × 25 × 30 × 52 |
| Spatial / temporal VAE compression | 16 / 4 | 16 / 4 |
| Latent patch size | 2 × 2 | 2 × 2 |
| RGB tokens per latent frame | 220 | 390 |
| Latent frames per chunk | 2 | 3 |
| New pixel frames per chunk | 8 | 12 |
| Generated chunks per request | 12 | 8 |
| Logical history capacity (latent frames) | 25: full request | 4: initial + 3 previous |
| Denoising steps per chunk | 4 SDE | 4 SDE |
| Denoising evaluations per request | 48 | 32 |
| Timesteps | 1, 15/16, 5/6, 5/8 | 1, 15/16, 5/6, 5/8 |
| Precision | BF16 | BF16 |
| Guidance / seed | 1 / 0 | 1 / 0 |
| Conditioning frame rate | 30 FPS | 30 FPS |
| Decoder | Full Wan VAE | Full Wan VAE |
| Denoising compile / CUDA graphs | Off / off | Off / off |
| Warmup / measured requests | 2 / 5 | 2 / 5 |

Each pair uses the same model revision and prepared image, prompt, controls and
other conditioning. Bimanual retains the complete request in both backends;
FlashDreams reserves 24 generated latent frames plus the initial frame, while
vLLM sizes its full-history cache to the 25-frame request. Auto retains the
initial frame and the previous three-frame chunk. Native attention, cache allocation and
decode scheduling can differ. Quality equivalence is unverified.

Generated FPS is 480 divided by the total time of five measured requests.
CUDA is synchronized around timing. Input encoding/tokenization, cache prefill,
generation, full VAE decoding and CPU uint8 output are timed. Weight loading,
input preparation and setup are excluded. Results on a shared GPU can vary.

The script checks the actual checkpoint chunk sizes and schedules and validates
paired settings and input/VAE hashes before publishing results. Bimanual also
checks the native encoder/decoder weights against the pinned vLLM VAE at BF16
before timing. Raw reports, logs,
inputs and environments stay ignored locally. Each MP4 is fully decoded and checked for 97 frames, 30 FPS and the expected
resolution; the report verifies its SHA256 before publishing. Bimanual prepends
its original conditioning image to the 96 generated frames; Auto/vLLM include
the first frame in their decoded output. Videos share playback duration and
seed, but matching seeds does not guarantee pixel equality between backends.
The website serves the comparison table, settings and these four generated
videos. `--build-only` exports a standalone `website/` folder with the videos.

Use `COSMOS3_AUTO_ENV` and `COSMOS3_OFFICIAL_ENV` consistently for setup and
benchmarking to select existing environments. Set `CUDA_HOME` for your CUDA
installation if needed. CPU check: `python3 benchmarks/cosmos3/check-three-seconds.py`.
