import os
import numpy as np
import torch
import mediapy as media
from einops import rearrange
from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig
from flashsim.model.video_vae.wan import WanVAEInterfaceConfig
from flashsim.model.video_dit.wan2_1.model import (
    WanDiTConfig,
    WanDiTNetwork1pt3BConfig,
    WanDiTCondition,
    NEGATIVE_PROMPT,
)

torch.manual_seed(42)
device = torch.device("cuda")
dtype = torch.bfloat16

text_encoder = WanTextEncoderConfig().setup(device=device)
vae = WanVAEInterfaceConfig().setup(device=device)

dit = WanDiTConfig(
    checkpoint_path="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/diffusion_pytorch_model.safetensors",
    network=WanDiTNetwork1pt3BConfig(),
    denoising_timesteps=list(range(1000, 0, -50)),
    warp_denoising_step=True,
    window_size_t=21,
    len_t=21,
).setup(device=device)

video_height = 480
video_width = 832
with torch.no_grad():
    TEXT_PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    positive_text_embeddings = text_encoder.encode([TEXT_PROMPT])  # [1, L, D]
    negative_text_embeddings = text_encoder.encode([NEGATIVE_PROMPT])  # [1, L, D]

    cache = dit.initialize_cache(
        height=video_height // vae.spatial_compression_ratio,
        width=video_width // vae.spatial_compression_ratio,
        positive_text_embeddings=positive_text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
    )
    cache.autoregressive_index = 0

    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    clean_latent = dit.generate(condition=WanDiTCondition(), cache=cache)
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    torch.cuda.synchronize()
    print(f"time on DiT: {start_event.elapsed_time(end_event)} ms")

    generated_video = vae.decode(clean_latent)
    print("Generated video shape:", generated_video.shape)

    # export result
    canvas = rearrange(generated_video, "1 t c h w -> t h w c")
    canvas = (canvas.float().cpu().numpy() + 1.0) / 2.0  # range [0, 1]
    canvas = (canvas * 255).astype(np.uint8)
    save_path = "outputs/wan2_1_t2v_1.3b.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_video(save_path, canvas, fps=16)
    print(f"saved generated video to {save_path}")
