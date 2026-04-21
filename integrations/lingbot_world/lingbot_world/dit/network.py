from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashsim.model.video_dit.wan2_1.modules import sinusoidal_embedding_1d
from flashsim.model.video_dit.wan2_1.network import (
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)

from .modules import CameraControlBlock


@dataclass
class LingbotWorldDiTNetworkCache(WanDiTNetworkCache):
    """Same block-cache container as WAN"""

    pass


@dataclass
class LingbotWorldDiTNetworkConfig(WanDiTNetworkConfig):
    """WAN-sized hyperparameters plus Lingbot camera / action control."""

    _target: type["LingbotWorldDiTNetwork"] = field(
        default_factory=lambda: LingbotWorldDiTNetwork
    )
    control_type: Literal["cam", "act"] = "cam"


@dataclass
class LingbotWorldDiTNetwork1pt3BConfig(LingbotWorldDiTNetworkConfig):
    """Configuration for the 1.3B Lingbot World DiT network."""

    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30


@dataclass
class LingbotWorldDiTNetwork14BConfig(LingbotWorldDiTNetworkConfig):
    """Configuration for the 14B Lingbot World DiT network."""

    dim: int = 5120
    ffn_dim: int = 13824
    num_heads: int = 40
    num_layers: int = 40


class LingbotWorldDiTNetwork(WanDiTNetwork):
    """Lingbot World DiT diffusion backbone for text-to-video and image-to-video."""

    def __init__(self, config: LingbotWorldDiTNetworkConfig) -> None:
        super().__init__(config)

        if config.control_type == "cam":
            control_dim = 6
        elif config.control_type == "act":
            control_dim = 7
        else:
            raise ValueError(f"Invalid control type: {config.control_type}")
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim
            * 64
            * self.patch_size[0]
            * self.patch_size[1]
            * self.patch_size[2],
            self.dim,
        )
        self.c2ws_hidden_states_layer1 = nn.Linear(self.dim, self.dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(self.dim, self.dim)

    def build_block(self, layer_idx: int) -> nn.Module:
        return CameraControlBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
        )

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        text_embeddings: Tensor,
    ) -> LingbotWorldDiTNetworkCache:
        """Initialize block caches from text context embeddings."""
        cache: WanDiTNetworkCache = super().initialize_cache(
            chunk_size, window_size, sink_size, text_embeddings, None
        )
        return LingbotWorldDiTNetworkCache(block_caches=cache.block_caches)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: LingbotWorldDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        plucker: Tensor | None = None,
    ) -> Tensor:
        """Run one denoising forward pass.

        Args:
            x: Input tokens of shape [..., L, D_in] after patchify.
                The layout is assumed to be
                "... (t h w) (c kt kh kw)".
            timesteps: Diffusion timesteps of shape [...].
            cache: Per-block KV caches.
            rope_freqs: RoPE frequencies of shape [L, 1, 1, head_dim // 2] after CP.
            current_chunk_idx: Current chunk index for streaming cache update.
            eager_mode: If True, run cache before/after update hooks.
            plucker: Optional Camera Control. Plucker embedding of shape
                [..., L, D], camera-to-world space.

        Returns:
            Tensor of shape [..., L, prod(patch_size) * out_dim].
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]

        if plucker is not None:
            plucker_embedding = self.patch_embedding_wancamctrl(plucker)
            plucker_hidden_states = self.c2ws_hidden_states_layer2(
                F.silu(self.c2ws_hidden_states_layer1(plucker_embedding))
            )
            plucker_embedding = plucker_embedding + plucker_hidden_states
        else:
            plucker_embedding = None

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)  # (..., L, D)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(
                self.dim, -1
            )  # [D, in_dim * kt * kh * kw]
            _bias = self.patch_embedding.bias  # [D] or None
            x = F.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )  # [..., D]
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))  # [..., 6, D]

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            x = block(
                x=x,
                e=torch.broadcast_to(e0, batch_shape + e0.shape[-2:]),
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                plucker_embedding=plucker_embedding,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        x = self.head(
            x, torch.broadcast_to(e, batch_shape + (1, e.shape[-1]))
        )  # (..., L, D)
        return x


# python -m lingbot_world.dit.network
if __name__ == "__main__":
    from flashsim.checkpoint.load import load_checkpoint

    network = LingbotWorldDiTNetwork14BConfig(
        control_type="cam",
        patch_embedding_type="conv3d",
        in_dim=16 + 20,  # i2v
    ).setup()

    checkpoint_path = "https://huggingface.co/robbyant/lingbot-world-fast/blob/main/diffusion_pytorch_model.safetensors.index.json"
    state_dict = load_checkpoint(checkpoint_path)
    network.load_state_dict(state_dict)
    network.update_parameters_after_loading_checkpoint()
