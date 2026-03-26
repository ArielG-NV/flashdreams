from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashsim.model.video_dit.base import BaseVideoDiT

@dataclass
class MockVideoDiTCache:
    """
    A mock cache for the video DiT.
    """
    autoregressive_index: int = -1


class MockVideoDiT(BaseVideoDiT[MockVideoDiTCache]):
    """
    A mock video DiT for testing purposes.
    """
    def __init__(self):
        super().__init__()
        
    def initialize_cache(self) -> MockVideoDiTCache:
        return MockVideoDiTCache()

    def predict_flow(self, noisy_input: Tensor, timestep: Tensor, condition: Any, cache: MockVideoDiTCache) -> Tensor:
        return torch.randn_like(noisy_input)