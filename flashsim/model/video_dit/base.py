from abc import ABC, abstractmethod

from typing import Any

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
    def predict_flow(self, noisy_input: Tensor, timestep: Tensor, condition: Any, cache: VideoDiTCacheType) -> Tensor:
        """
        Predict the flow for denoising the input tensor.

        Args:
            noisy_input: The noisy input tensor. [B, ...]
            timestep: The timestep. [B]
            condition: The condition.
            cache: The autoregressive cache to use for the DIT.

        Returns:
            The predicted flow. Same shape as the input tensor `noisy_input`.
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

    def denoise(self, noisy_input: Tensor, timestep: Tensor, predicted_flow: Tensor) -> Tensor:
        """
        Recover the clean input from the noisy input.

        Args:
            noisy_input: The noisy input tensor. [B, ...]
            timestep: The timestep. [B]
            predicted_flow: The predicted flow. Same shape as the input tensor `noisy_input`.

        Returns:
            The clean input tensor. Same shape as the input tensor `noisy_input`.
        """
        sigma = self.timestep_to_sigma(timestep) # [B]
        sigma = sigma.view(-1, *([1] * (len(noisy_input.shape) - 1))) # [B, ...]
        clean_input = noisy_input - sigma * predicted_flow
        return clean_input

    def add_noise(self, clean_input: Tensor, timestep: Tensor, rng: torch.Generator | None = None) -> Tensor:
        """
        Add noise to the clean input.

        Args:
            clean_input: The clean input tensor. [B, ...]
            timestep: The timestep. [B]
            rng: The random number generator to use for the noise.

        Returns:
            The noisy input tensor. Same shape as the input tensor `clean_input`.
        """
        sigma = self.timestep_to_sigma(timestep) # [B]
        sigma = sigma.view(-1, *([1] * (len(clean_input.shape) - 1))) # [B, ...]
        noise = torch.randn_like(clean_input, generator=rng)
        noisy_input = (1.0 - sigma) * clean_input + sigma * noise
        return noisy_input