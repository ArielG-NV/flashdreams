import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashsim.model.video_dit.wan2_1.modules import (
    Block,
    BlockCache,
)


class CameraControlBlock(Block):
    """Transformer block with camera control."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        plucker_embedding: Tensor | None = None,
    ) -> Tensor:
        """Run one transformer block update.

        Args:
            x: Input tensor with shape [..., L, D].
            e: Modulation tensor with shape [..., 6, D].
            cache: KV cache container for this block.
            rope_freqs: RoPE frequencies with shape [L, 1, 1, head_dim // 2].
            plucker_embedding: Optional Camera Control. Plucker embedding of
                shape [..., L, D], camera-to-world space.

        Returns:
            Updated hidden states with shape [..., L, D].
        """
        e = (self.modulation + e).chunk(6, dim=-2)  # [..., 1, D] each

        y = self.norm1(x) * (1 + e[1]) + e[0]  # [..., L, D]
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = x + (y * e[2])  # [..., L, D]

        if plucker_embedding is not None:
            camera_hidden_states = self.cam_injector_layer2(
                F.silu(self.cam_injector_layer1(plucker_embedding))
            )
            camera_hidden_states = camera_hidden_states + plucker_embedding
            camera_scale = self.cam_scale_layer(camera_hidden_states)
            camera_shift = self.cam_shift_layer(camera_hidden_states)
            x = (1.0 + camera_scale) * x + camera_shift

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e[4]) + e[3]  # [..., L, D]
        y = self.ffn(y)
        x = x + (y * e[5])  # [..., L, D]
        return x
