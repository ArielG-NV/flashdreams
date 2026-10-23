# Cosmos3 results

GPU: NVIDIA GB300

90 frames at 30 FPS (3.000 seconds), 97 work frames, four denoising steps, guidance 1, full Wan VAE. Two warmups and five timed requests; encoding/tokenization, generation, decoding and CPU uint8 output are timed. Weight loading and MP4 export are excluded.

| Model | Backend | Resolution | Steps | Median request | Generated FPS |
| --- | --- | --- | ---: | ---: | ---: |
| base Nano | vllm | 832x480 | 4 | 4.2753 s | 22.49 |
| bimanual | flashdreams | 640x352 | 4 | 2.8678 s | 33.66 |
| bimanual | vllm | 640x352 | 4 | 6.6906 s | 14.32 |
| auto | flashdreams | 832x480 | 4 | 7.1495 s | 13.45 |
| auto | vllm | 832x480 | 4 | 17.3183 s | 5.51 |

Each model pair shares resolution, duration, steps, seed and prepared inputs. Bimanual uses six four-latent-frame chunks; AV uses eight three-latent-frame chunks with the initial frame and one previous chunk retained. Caption/control prefill and numerical kernels can differ; quality equivalence is unverified. Base Nano is not distilled and can have poor quality at four steps. Measurements were collected on a shared GPU.

Reproduce: `bash benchmarks/cosmos3/run.sh`. View: `python benchmarks/cosmos3/results-site.py`.
