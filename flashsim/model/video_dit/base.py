from abc import ABC, abstractmethod

from typing import Any

from einops import rearrange

import torch
from torch import Tensor


class BaseVideoDiT[VideoDiTCacheType](ABC):

    @abstractmethod
    def initialize_cache(self) -> VideoDiTCacheType:
        """
        Initialize the cache for DIT.
        """
        ...

    @abstractmethod
    def predict_x0(
        self, 
        x0: Tensor | None, # clean latent
        timestep: Tensor, 
        condition: Any, 
        cache: VideoDiTCacheType, 
        rng: torch.Generator | None = None
    ) -> Tensor:
        """
        Predict the flow for denoising the input tensor.

        Args:
            x0: The clean latent. [B, ...]
            timestep: The timestep. [1] or [B]
            condition: The condition.
            cache: The autoregressive cache to use for the DIT.
            rng: The random number generator to use for the noise.

        Returns:
            The predicted clean latent. Same shape as the input tensor `x0`.
        """
        ...

    @abstractmethod
    def timestep_to_sigma(self, timestep: Tensor) -> Tensor:
        """
        Convert the timestep to the sigma.

        Args:
            timestep: The timestep. [B]

        Returns:
            The sigma. [B]
        """
        ...

    @property
    @abstractmethod
    def temporal_patch_size(self) -> int:
        """
        Get the temporal patch size.
        """
        ...

    @property
    @abstractmethod
    def spatial_patch_size(self) -> int:
        """
        Get the spatial patch size.
        """
        ...

    @property
    @abstractmethod
    def denoising_timesteps(self) -> list[int]:
        """
        Get the denoising timesteps.
        """
        ...

    def denoise(self, noisy_input: Tensor, timestep: Tensor, predicted_flow: Tensor) -> Tensor:
        """
        Recover the clean input from the noisy input.

        Args:
            noisy_input: The noisy input tensor. [B, ...]
            timestep: The timestep. [1] or [B]
            predicted_flow: The predicted flow. Same shape as the input tensor `noisy_input`.

        Returns:
            The clean input tensor. Same shape as the input tensor `noisy_input`.
        """
        sigma = self.timestep_to_sigma(timestep) # [B]
        sigma = sigma.view(-1, *([1] * (len(noisy_input.shape) - 1))) # [1, ...] or [B, ...]
        clean_input = noisy_input - sigma * predicted_flow
        return clean_input

    def add_noise(self, clean_input: Tensor, timestep: Tensor, rng: torch.Generator | None = None) -> Tensor:
        """
        Add noise to the clean input.

        Args:
            clean_input: The clean input tensor. [B, ...]
            timestep: The timestep. [1] or [B]
            rng: The random number generator to use for the noise.

        Returns:
            The noisy input tensor. Same shape as the input tensor `clean_input`.
        """
        sigma = self.timestep_to_sigma(timestep) # [B]
        sigma = sigma.view(-1, *([1] * (len(clean_input.shape) - 1))) # [1, ...] or [B, ...]
        noise = torch.randn_like(clean_input, generator=rng)
        noisy_input = (1.0 - sigma) * clean_input + sigma * noise
        return noisy_input

    def patchify(self, x: Tensor) -> Tensor:
        """
        Patchify the input tensor.

        The patchify pattern is:
            "... c (t kt) (h kh) (w kw) -> ... t h w (c kt kh kw)"

        Args:
            x: The input tensor. [..., C, T, H, W]

        Returns:
            The patched tensor. [..., len_t, len_h, len_w, D]
        """
        x = rearrange(
            x,
            "... c (t kt) (h kh) (w kw) -> ... t h w (c kt kh kw)",
            kt=self.temporal_patch_size,
            kh=self.spatial_patch_size,
            kw=self.spatial_patch_size,
        )
        return x
    
    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Unpatchify the input tensor.

        The unpatchify pattern is:
            "... t h w (c kt kh kw) -> ... c (t kt) (h kh) (w kw)"

        Args:
            x: The input tensor. [..., len_t, len_h, len_w, D]

        Returns:
            The unpatched tensor. [..., C, T, H, W]
        """
        x = rearrange(
            x,
            "... t h w (c kt kh kw) -> ... c (t kt) (h kh) (w kw)",
            kt=self.temporal_patch_size,
            kh=self.spatial_patch_size,
            kw=self.spatial_patch_size,
        )
        return x

    def generate(
        self, 
        condition: Any, 
        cache: VideoDiTCacheType, 
        rng: torch.Generator | None = None
    ) -> Tensor:
        x0 = None # clean latent
        for denoising_step in self.denoising_timesteps:
            timestep = torch.tensor([denoising_step], device=self.device, dtype=self.dtype)
            x0 = self.predict_x0(x0, timestep, condition, cache, rng=rng)
        return x0
