from lingbot_world.pipeline import LingbotWorldPipelineConfig
from flashsim.model.video_vae.wan import (
    WanVAEInterfaceConfig,
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
)
from flashsim.model.video_vae.teahv import (
    TeahvInterfaceConfig,
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
)
from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig
from lingbot_world.dit.model import (
    LingbotWorldDiTConfig,
    AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS,
)
from lingbot_world.dit.network import (
    LingbotWorldDiTNetwork14BConfig,
)
from flashsim.model.video_vae.pshuffle import PixelShuffleVAEInterfaceConfig

LINGBOT_WORLD_CONFIGS = {}

LINGBOT_WORLD_CONFIGS["LingBot-World-Fast"] = LingbotWorldPipelineConfig(
    tokenizer=PixelShuffleVAEInterfaceConfig(),
    detokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    text_encoder=WanTextEncoderConfig(),
    dit=LingbotWorldDiTConfig(
        checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS["LingBot-World-Fast"],
        network=LingbotWorldDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
            control_type="cam",
            in_dim=16 + 20,  # i2v
        ),
    ),
)

LINGBOT_WORLD_CONFIGS["LingBot-World-Fast-Flash"] = LingbotWorldPipelineConfig(
    tokenizer=PixelShuffleVAEInterfaceConfig(),
    detokenizer=TeahvInterfaceConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
    ),
    text_encoder=WanTextEncoderConfig(),
    dit=LingbotWorldDiTConfig(
        checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS["LingBot-World-Fast"],
        network=LingbotWorldDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
            control_type="cam",
            in_dim=16 + 20,  # i2v
        ),
        # denoising_timesteps=[999, 978, 947, 825],
        denoising_timesteps=[999, 947],
    ),
)
