# Cosmos3: FlashDreams vs. vLLM

From the repository root on a CUDA Linux host with `uv` and model/dataset access:

1. Set up the environments:

   ```bash
   bash benchmarks/cosmos3/setup.sh flashdreams
   bash benchmarks/cosmos3/setup.sh auto
   bash benchmarks/cosmos3/setup.sh vllm
   ```

2. Run all four benchmarks and print the GPU/FPS table:

   ```bash
   bash benchmarks/cosmos3/run.sh --skip-setup
   ```

   Inputs are prepared automatically. The table is saved to [RESULTS.md](RESULTS.md).
   For setup and benchmarking in one command, use `bash benchmarks/cosmos3/run.sh`.

3. Start the results website, then open <http://localhost:8765>:

   ```bash
   python3 benchmarks/cosmos3/results-site.py
   ```

Both backends use these settings within each model pair. Latent dimensions
are channels × time × height × width; history capacity includes the initial frame.

| Setting | Bimanual: both backends | Auto: both backends |
| --- | --- | --- |
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
inputs and environments stay ignored locally. The website shows only the GPU,
FPS table and settings table; it does not serve raw reports or input files.

Use `COSMOS3_AUTO_ENV` and `COSMOS3_OFFICIAL_ENV` consistently for setup and
benchmarking to select existing environments. Set `CUDA_HOME` for your CUDA
installation if needed. CPU check: `python3 benchmarks/cosmos3/check-three-seconds.py`.
