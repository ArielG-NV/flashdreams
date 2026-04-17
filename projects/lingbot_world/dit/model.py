from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashsim.checkpoint.load import load_checkpoint
from flashsim.config import InstantiateConfig

from flashsim.model.video_dit.base import BaseVideoDiT, denoise, add_noise
from flashsim.model.video_dit.rope import RotaryPositionEmbedding3D
from flashsim.model.video_dit.flow_match import FlowMatchScheduler
from flashsim.model.video_dit.context_parallel_strategy import (
    HierarchicalCPGroups,
    create_hierarchical_cp_groups,
)
from .network import (
    LingbotWorldDiTNetwork,
    LingbotWorldDiTNetworkCache,
    LingbotWorldDiTNetwork14BConfig,
)


AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS = {
    "LingBot-World-Fast": "https://huggingface.co/robbyant/lingbot-world-fast/blob/main/diffusion_pytorch_model.safetensors.index.json",
}


@dataclass
class LingbotWorldDiTCondition:
    """
    Condition for the Lingbot World DiT.
    """

    plucker: Tensor  # [..., T, C, H, W]

    _is_patchified: bool = False


@dataclass
class LingbotWorldDiTCache:
    """
    Cache for the Wan DiT.
    """

    len_h: (
        int  # number of tokens along the spatial height dimension after patchification
    )
    len_w: (
        int  # number of tokens along the spatial width dimension after patchification
    )
    num_tokens_per_chunk: int  # number of tokens per chunk after CP
    batch_shape: tuple[int, ...]  # The batch shape for (...)

    condition_image: Tensor  # mask and image latents [..., T, 4+C, H, W]

    network_cache_conditioned: LingbotWorldDiTNetworkCache
    network_cache_unconditioned: LingbotWorldDiTNetworkCache | None
    rope_adapter: RotaryPositionEmbedding3D

    # For KV cache update in the end.
    x0: Tensor | None = None  # clean latent [..., pTHW, D]
    condition: LingbotWorldDiTCondition | None = None

    autoregressive_index: int = -1
    _is_patchified: bool = False


@dataclass
class LingbotWorldDiTConfig(InstantiateConfig["LingbotWorldDiT"]):
    _target: type["LingbotWorldDiT"] = field(default_factory=lambda: LingbotWorldDiT)

    # Network configurations
    network: LingbotWorldDiTNetwork14BConfig = field(
        default_factory=lambda: LingbotWorldDiTNetwork14BConfig()
    )
    dtype: torch.dtype = torch.bfloat16

    # RoPE: Default to 1.0 for no extrapolation.
    h_extrapolation_ratio: float = 1.0
    w_extrapolation_ratio: float = 1.0

    # Difussion schedule
    denoising_timesteps: list[int] = field(default_factory=lambda: [999, 978, 947, 825])
    warp_denoising_step: bool = False

    # Local attn: Number of tokens along T dimension.
    window_size_t: int = 60  # official code uses global attn (no sliding window)
    sink_size_t: int = 0

    # Chunk size: Number of tokens along T dimension. (after patchification)
    len_t: int = 3

    # Checkpoint path
    checkpoint_path: str | None = None

    # Noise level for KV cache update.
    context_noise: int = 0

    # Speedup.
    compile_network: bool = True

    # CFG guidance scale.
    guidance_scale: float = 1.0
    shift: float = 8.0


class LingbotWorldDiT(BaseVideoDiT[LingbotWorldDiTCache]):
    """
    Lingbot World DiT for video generation.
    """

    def __init__(
        self, config: LingbotWorldDiTConfig, device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        # multi-GPU setup
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self.cp_groups = create_hierarchical_cp_groups(
                world_size=world_size,
                rank=rank,
                V=1,
                T=config.len_t,
                single_group_as_none=True,
            )
        else:
            self.cp_groups = HierarchicalCPGroups(rank=0)

        self.config = config
        self.dtype = config.dtype
        self.device = device

        self.network = LingbotWorldDiTNetwork(config=self.config.network)
        self.network = self.network.to(device=self.device, dtype=self.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(
            cp_group=self.cp_groups.THW_group,
        )

        if self.config.checkpoint_path is not None:
            state_dict = load_checkpoint(self.config.checkpoint_path)
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

        if self.config.compile_network:
            self.network = torch.compile(
                self.network, mode="max-autotune-no-cudagraphs"
            )

        # define scheduler
        num_train_timestep = 1000
        self.scheduler = FlowMatchScheduler(
            shift=self.config.shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(num_train_timestep, training=True)
        if self.config.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[
                num_train_timestep
                - torch.tensor(self.config.denoising_timesteps, dtype=torch.long)
            ]
        else:
            self.denoising_step_list = torch.tensor(
                self.config.denoising_timesteps, dtype=torch.long
            )
        self.denoising_step_list = self.denoising_step_list.to(self.device, self.dtype)

    def initialize_cache(
        self,
        height: int,
        width: int,
        positive_text_embeddings: Tensor,  # [..., L, D]
        negative_text_embeddings: Tensor | None = None,  # [..., L, D]
        image_embeddings: Tensor | None = None,  # [..., L, D]
        image_latents: Tensor | None = None,  # [..., T, C, H, W]
    ) -> LingbotWorldDiTNetworkCache:
        """
        Initialize the cache for the video DiT.

        Args:
            height: The video height after VAE spatial compression.
            width: The video width after VAE spatial compression.
            positive_text_embeddings: Positive text embeddings [..., L, D]
            negative_text_embeddings: Negative text embeddings [..., L, D]
            image_embeddings: CLIP Image embeddings [..., L, D]
            image_latent: VAE encoded image latent [..., T, C, H, W]

        Returns:
            The cache for the video DiT.
        """
        # compute size of the tokens after patchification
        len_t = self.config.len_t
        len_h = height // self.config.network.patch_size[1]
        len_w = width // self.config.network.patch_size[2]

        head_dim = self.config.network.dim // self.config.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=len_t,
            len_h=len_h,
            len_w=len_w,
            head_dim=head_dim,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            interleaved=True,
            device=self.device,
        )
        # RoPE CP splits along same dimension as self-attention CP.
        rope_adapter.set_context_parallel_group(cp_group=self.cp_groups.THW_group)

        num_tokens_per_chunk = len_t * len_h * len_w
        num_tokens_window_size = self.config.window_size_t * len_h * len_w
        num_tokens_sink_size = self.config.sink_size_t * len_h * len_w
        if self.cp_groups.THW_group is not None:
            num_tokens_per_chunk //= self.cp_groups.THW_group.size()
            num_tokens_window_size //= self.cp_groups.THW_group.size()
            num_tokens_sink_size //= self.cp_groups.THW_group.size()
        network_cache_conditioned = self.network.initialize_cache(
            chunk_size=num_tokens_per_chunk,
            window_size=num_tokens_window_size,
            sink_size=num_tokens_sink_size,
            text_embeddings=positive_text_embeddings,
        )
        # CFG guidance
        if negative_text_embeddings is not None:
            network_cache_unconditioned = self.network.initialize_cache(
                chunk_size=num_tokens_per_chunk,
                window_size=num_tokens_window_size,
                sink_size=num_tokens_sink_size,
                text_embeddings=negative_text_embeddings,
            )
        else:
            network_cache_unconditioned = None

        *batch_shape, T, _, H, W = image_latents.shape
        image_masks = torch.zeros(
            *batch_shape, T, 4, H, W, device=self.device, dtype=self.dtype
        )
        image_masks[..., :1, :, :, :] = 1.0
        condition_image = torch.cat([image_masks, image_latents], dim=-3)

        cache = LingbotWorldDiTCache(
            len_h=len_h,
            len_w=len_w,
            network_cache_conditioned=network_cache_conditioned,
            network_cache_unconditioned=network_cache_unconditioned,
            rope_adapter=rope_adapter,
            num_tokens_per_chunk=num_tokens_per_chunk,
            batch_shape=positive_text_embeddings.shape[:-2],
            condition_image=condition_image,
        )
        cache = self._patchify(cache)
        return cache

    def generate(
        self,
        condition: LingbotWorldDiTCondition,
        cache: LingbotWorldDiTCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        condition = self._patchify(condition)
        x0 = None  # clean latent
        for denoising_step in self.denoising_step_list:
            timestep = torch.tensor(
                [denoising_step], device=self.device, dtype=self.dtype
            )
            x0 = self._predict_x0(x0, timestep, condition, cache, rng=rng)

        # Postpone KV cache update to the finalization step.
        cache.x0 = x0
        cache.condition = condition

        x0 = self._unpatchify(cache.len_h, cache.len_w, x0)
        return x0

    def finalize(
        self,
        cache: LingbotWorldDiTCache,
        context_noise: int | None = None,
        rng: torch.Generator | None = None,
    ) -> None:
        # update kv cache
        if context_noise is None:
            context_noise = self.config.context_noise
        timestep = torch.tensor([context_noise], device=self.device, dtype=self.dtype)
        _ = self._predict_x0(cache.x0, timestep, cache.condition, cache, rng=rng)

    def _predict_x0(
        self,
        x0: Tensor | None,  # clean latent [..., pT, pHW, D]
        timestep: Tensor,  # [1]
        condition: LingbotWorldDiTCondition,
        cache: LingbotWorldDiTNetworkCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before predicting flow"
        assert timestep.shape == (1,), "Timestep must be a scalar shape"
        alpha = self.scheduler.timestep_to_sigma(timestep)

        rope_freqs = cache.rope_adapter.shift_t(
            offset=autoregressive_index * self.config.len_t
        )
        batch_shape = cache.batch_shape
        len_thw = cache.num_tokens_per_chunk

        token_dim = (
            self.config.network.out_dim
            * self.config.network.patch_size[0]
            * self.config.network.patch_size[1]
            * self.config.network.patch_size[2]
        )
        input_shape = (*batch_shape, len_thw, token_dim)

        if x0 is None:
            # pure noise
            noisy_input = torch.randn(
                input_shape, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            noisy_input = add_noise(x0, alpha, rng=rng)

        assert noisy_input.shape == input_shape

        # I2V
        thw_start = autoregressive_index * len_thw
        thw_end = thw_start + len_thw
        if autoregressive_index == 0:
            y = cache.condition_image[..., thw_start:thw_end, :]
        else:
            if thw_end <= cache.condition_image.shape[-2]:
                y = cache.condition_image[..., thw_start:thw_end, :]
            else:
                y = cache.condition_image[..., -len_thw:, :]

        predicted_flow_conditioned = self.network(
            x=torch.cat([noisy_input, y], dim=-1),
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=cache.network_cache_conditioned,
            current_chunk_idx=autoregressive_index,
            eager_mode=True,
            plucker=condition.plucker,
        )
        # CFG guidance
        if (
            cache.network_cache_unconditioned is not None
            and self.config.guidance_scale > 1.0
        ):
            predicted_flow_unconditioned = self.network(
                x=torch.cat([noisy_input, y], dim=-1),
                timesteps=timestep,
                rope_freqs=rope_freqs,
                cache=cache.network_cache_unconditioned,
                current_chunk_idx=autoregressive_index,
                eager_mode=True,
                plucker=condition.plucker,
            )
            predicted_flow = (
                predicted_flow_unconditioned
                + self.config.guidance_scale
                * (predicted_flow_conditioned - predicted_flow_unconditioned)
            )
        else:
            predicted_flow = predicted_flow_conditioned

        x0 = denoise(noisy_input, alpha, predicted_flow)

        return x0

    def _patchify(
        self, x: Tensor | LingbotWorldDiTCondition | LingbotWorldDiTCache
    ) -> Tensor:
        process_groups = [
            self.cp_groups.THW_group,
        ]
        cp_dims = [-2]

        if isinstance(x, LingbotWorldDiTCache):
            if x._is_patchified:
                return x
            else:
                # x.condition_image stores [..., T*num_blocks, 4+C, H, W]
                per_block_images = torch.split(
                    x.condition_image,
                    dim=-4,
                    split_size_or_sections=self.config.len_t,
                )
                per_block_images = [
                    self.network.patchify_and_maybe_split_cp(
                        image,
                        process_groups=process_groups,
                        cp_dims=cp_dims,
                    )
                    for image in per_block_images
                ]
                x.condition_image = torch.cat(per_block_images, dim=-2)
                x._is_patchified = True
                return x
        if isinstance(x, LingbotWorldDiTCondition):
            if x._is_patchified:
                return x
            else:
                x.plucker = self.network.patchify_and_maybe_split_cp(
                    x.plucker,
                    process_groups=process_groups,
                    cp_dims=cp_dims,
                )
                x._is_patchified = True
                return x
        elif isinstance(x, Tensor):
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=process_groups,
                cp_dims=cp_dims,
            )
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    def _unpatchify(self, len_h: int, len_w: int, x: Tensor) -> Tensor:
        process_groups = [
            self.cp_groups.THW_group,
        ]
        cp_dims = [-2]

        return self.network.unpatchify_and_maybe_gather_cp(
            pH=len_h,
            pW=len_w,
            x=x,
            process_groups=process_groups,
            cp_dims=cp_dims,
        )
