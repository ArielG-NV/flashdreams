import math

import torch
import torch.nn as nn
from torch import Tensor


class GPT2FeedForward(nn.Module):
    """GPT-2 style feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (..., D).

        Returns:
            Output tensor of shape (..., D).
        """
        return self.layer2(self.activation(self.layer1(x)))


class Timesteps(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    SINUSOIDAL_FREQ_BASE = 10000

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

        half_dim = num_channels // 2
        exponent = -math.log(self.SINUSOIDAL_FREQ_BASE) * torch.arange(half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed timesteps into sinusoidal frequencies.

        Args:
            timesteps: Input tensor of shape (...).

        Returns:
            Embedded tensor of shape (..., num_channels).
        """
        in_dtype = timesteps.dtype
        emb = timesteps.unsqueeze(-1).float() * self.emb
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.to(dtype=in_dtype)


class TimestepEmbedding(nn.Module):
    """MLP for encoding timestep embeddings with optional AdaLN-LoRA."""

    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = True) -> None:
        super().__init__()
        self.use_adaln_lora = use_adaln_lora

        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()

        out_dim = 3 * out_features if use_adaln_lora else out_features
        self.linear_2 = nn.Linear(out_features, out_dim, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Encode timestep embedding.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tuple of (emb, adaln_lora):
                - emb: Output tensor of shape (..., out_features).
                - adaln_lora: If use_adaln_lora, tensor of shape (..., 3 * out_features); otherwise None.
        """
        out = self.linear_2(self.activation(self.linear_1(x)))

        if self.use_adaln_lora:
            return x, out
        return out, None


class PatchEmbed(nn.Module):
    """Patch embedding module for video/image inputs.

    Note: The patchify operation (rearranging from spatial to patch tokens) is expected
    to be performed externally. This module expects post-patchified flattend input of shape (..., D)
    where D = in_channels * temporal_patch_size * spatial_patch_size^2.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ) -> None:
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels

        self.proj = nn.Sequential(
            nn.Identity(),  # Placeholder for checkpoint compatibility
            nn.Linear(self._compute_in_features(), out_channels, bias=False),
        )

    def _compute_in_features(self) -> int:
        """Compute the flattened patch dimension."""
        return self.in_channels * self.temporal_patch_size * self.spatial_patch_size**2

    def get_linear_in_channels(self) -> int:
        """Return input dimension for the linear projection (for external use)."""
        return self._compute_in_features()

    def forward(self, x: Tensor) -> Tensor:
        """Project flattened patches to embedding space.

        Args:
            x: Input tensor of shape (..., D) where D = in_channels * kt * kh * kw.

        Returns:
            Embedded patches of shape (..., out_channels).
        """
        expected_in_features = self._compute_in_features()
        assert x.shape[-1] == expected_in_features, (
            f"Expected input features to be {expected_in_features}, but got {x.shape[-1]}."
        )
        return self.proj(x)


class FinalLayer(nn.Module):
    """Final layer of the DiT network with AdaLN modulation."""

    NUM_ADALN_CHUNKS = 2

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        patch_dim = spatial_patch_size**2 * temporal_patch_size * out_channels
        self.linear = nn.Linear(hidden_size, patch_dim, bias=False)

        modulation_out_dim = self.NUM_ADALN_CHUNKS * hidden_size
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, modulation_out_dim, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, modulation_out_dim, bias=False),
            )

    def forward(self, x: Tensor, emb: Tensor, adaln_lora: Tensor | None = None) -> Tensor:
        """Apply final layer with adaptive layer normalization.

        Args:
            x: Input tensor of shape (B, ..., D).
            emb: Conditioning embedding of shape (B, D).
            adaln_lora: Optional LoRA tensor of shape (B, 3 * D).

        Returns:
            Output tensor of shape (B, ..., D') where D' = patch_dim.
        """
        batch_size, *ellipsis_dims, hidden_dim = x.shape
        assert emb.shape == (batch_size, hidden_dim)

        emb = emb.reshape(batch_size, *([1] * len(ellipsis_dims)), hidden_dim)

        if self.use_adaln_lora:
            assert adaln_lora is not None and adaln_lora.shape == (batch_size, 3 * hidden_dim)
            adaln_lora = adaln_lora.reshape(batch_size, *([1] * len(ellipsis_dims)), 3 * hidden_dim)
            modulation = self.adaln_modulation(emb) + adaln_lora[..., : 2 * self.hidden_size]
            shift, scale = modulation.chunk(2, dim=-1)
        else:
            shift, scale = self.adaln_modulation(emb).chunk(2, dim=-1)

        x = self.layer_norm(x) * (1.0 + scale) + shift
        return self.linear(x)
