from dataclasses import dataclass, field

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from flashsim.model.video_vae.wan import WanVAEInterfaceConfig, WanVAECache
from flashsim.model.video_vae.teahv import TeahvInterfaceConfig, TAEHVCache
from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig
from projects.lingbot_world.dit.model import (
    LingbotWorldDiTCache,
    LingbotWorldDiTCondition,
    LingbotWorldDiTConfig,
)
from flashsim.config import InstantiateConfig
from flashsim.model.video_dit.profiling import ProfileEvents

from .camera_utils import (
    compute_relative_poses_causal,
    get_plucker_embeddings,
)


@dataclass
class LingbotWorldPipelineCache:
    tokenizer_cache: WanVAECache | TAEHVCache
    detokenizer_cache: WanVAECache | TAEHVCache
    dit_cache: LingbotWorldDiTCache
    profile_events: list[ProfileEvents]


@dataclass
class LingbotWorldPipelineConfig(InstantiateConfig["LingbotWorldPipeline"]):
    _target: type["LingbotWorldPipeline"] = field(
        default_factory=lambda: LingbotWorldPipeline
    )

    tokenizer: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: WanVAEInterfaceConfig()
    )
    detokenizer: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: TeahvInterfaceConfig()
    )
    text_encoder: WanTextEncoderConfig = field(
        default_factory=lambda: WanTextEncoderConfig()
    )
    image_encoder: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: WanVAEInterfaceConfig()
    )
    dit: LingbotWorldDiTConfig = field(default_factory=lambda: LingbotWorldDiTConfig())

    seed: int = 42


class LingbotWorldPipeline:
    def __init__(
        self,
        config: LingbotWorldPipelineConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.text_encoder = config.text_encoder.setup(device=device)
        self.image_encoder = config.image_encoder.setup(device=device)
        self.tokenizer = config.tokenizer.setup(device=device)
        self.detokenizer = config.detokenizer.setup(device=device)
        self.dit = config.dit.setup(device=device)
        self.rng = torch.Generator(device=device).manual_seed(config.seed)

        self.last_pose: Tensor | None = None

    def initialize_cache(
        self, text: list[list[str]], image: Tensor
    ) -> LingbotWorldPipelineCache:
        """
        Initialize the cache for the Alpadreams pipeline.

        Args:
            text: The batch of texts to encode. [B, V]
            image: The first frame of the video. [B, V, 1, 3, H, W]
        """
        video_height, video_width = image.shape[-2:]

        encoded_height = video_height // self.tokenizer.spatial_compression_ratio
        encoded_width = video_width // self.tokenizer.spatial_compression_ratio

        image_padded = F.pad(image, (0, 0, 0, 0, 0, 0, 0, 81 - 1))
        image_latents = self.image_encoder.encode(image_padded)
        text_embeddings = torch.stack(
            [self.text_encoder.encode(t) for t in text], dim=0
        )

        dit_cache = self.dit.initialize_cache(
            height=encoded_height,
            width=encoded_width,
            positive_text_embeddings=text_embeddings,
            image_latents=image_latents,
        )

        tokenizer_cache = self.tokenizer.initialize_encode_cache()
        detokenizer_cache = self.detokenizer.initialize_decode_cache()

        return LingbotWorldPipelineCache(
            tokenizer_cache=tokenizer_cache,
            detokenizer_cache=detokenizer_cache,
            dit_cache=dit_cache,
            profile_events=[],
        )

    @torch.no_grad()
    def streaming_inference(
        self,
        autoregressive_index: int,
        height: int,
        width: int,
        intrinsics: Tensor,
        poses: Tensor,
        cache: LingbotWorldPipelineCache,
        world_scale: float = 1.0,
    ) -> Tensor:
        """
        Stream the inference of the video diffusion pipeline.

        Args:
            autoregressive_index: The autoregressive index.
            intrinsics: The camera intrinsics. [B, V, T, 4]
            poses: The camera-to-world poses. [B, V, T, 4, 4]
            cache: The cache for the Alpadreams pipeline.

        Returns:
            The decoded video. [B, V, T, C, H, W]
        """
        if autoregressive_index >= len(cache.profile_events):
            cache.profile_events.append(ProfileEvents())
        profile_events = cache.profile_events[autoregressive_index]

        if profile_events is not None:
            profile_events.tic.record()

        # 1. encode the hdmap
        plucker = self.render_plucker(
            height=height,
            width=width,
            intrinsics=intrinsics,
            poses=poses,
            world_scale=world_scale,
        )
        if hasattr(cache.tokenizer_cache, "autoregressive_index"):
            cache.tokenizer_cache.autoregressive_index = autoregressive_index
        encoded_plucker = self.tokenizer.encode(plucker, cache=cache.tokenizer_cache)

        if profile_events is not None:
            profile_events.toc_after_encode.record()

        # 2. run DiT denoising
        cache.dit_cache.autoregressive_index = autoregressive_index
        clean_input = self.dit.generate(
            condition=LingbotWorldDiTCondition(plucker=encoded_plucker),
            cache=cache.dit_cache,
            rng=self.rng,
        )

        if profile_events is not None:
            profile_events.toc_after_denoise.record()

        # 3. decode the clean input
        if hasattr(cache.detokenizer_cache, "autoregressive_index"):
            cache.detokenizer_cache.autoregressive_index = autoregressive_index
        decoded_video = self.detokenizer.decode(
            clean_input, cache=cache.detokenizer_cache
        )

        if profile_events is not None:
            profile_events.toc_after_decode.record()

        return decoded_video

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: LingbotWorldPipelineCache,
    ) -> None:
        """
        Finalize the streaming inference. This will update the KV cache for the next block.
        """
        self.dit.finalize(cache.dit_cache, rng=self.rng)

        profile_events = cache.profile_events[autoregressive_index]
        profile_events.toc_after_finalize.record()

    @torch.no_grad()
    def get_num_frames(self, autoregressive_index: int) -> int:
        """
        Get the number of frames for the given autoregressive index.
        """
        if autoregressive_index == 0:
            return (
                1
                + (self.dit.config.len_t - 1)
                * self.detokenizer.temporal_compression_ratio
            )
        else:
            return self.dit.config.len_t * self.detokenizer.temporal_compression_ratio

    @torch.no_grad()
    def render_plucker(
        self,
        height: int,
        width: int,
        intrinsics: Tensor,
        poses: Tensor,
        world_scale: float = 1.0,
    ) -> Tensor:
        """
        Encode the plucker embeddings for the given intrinsics and poses.

        Args:
            intrinsics: The camera intrinsics. [..., 4]
            poses: The camera-to-world poses. [..., 4, 4]
            world_scale: The world scale used to normalize the poses.

        Returns:
            The plucker embeddings. [..., C, H, W]
        """
        assert intrinsics.dtype == poses.dtype == torch.float32
        *batch_shape, _4, _4 = poses.shape
        batch_size = math.prod(batch_shape)
        intrinsics = intrinsics.view(batch_size, 4)
        poses = poses.view(batch_size, 4, 4)

        relative_poses = compute_relative_poses_causal(
            poses, world_scale, ref_pose=self.last_pose
        )
        plucker = get_plucker_embeddings(relative_poses, intrinsics, height, width)
        plucker = rearrange(plucker, "b h w c -> b c h w").to(torch.bfloat16)
        plucker = plucker.reshape(*batch_shape, *plucker.shape[-3:])

        self.last_pose = poses[..., -1:, :, :]
        return plucker
