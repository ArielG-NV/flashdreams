import torch
from torch import Tensor

from flashsim.model.text_encoder.base import BaseTextEncoder

class MockTextEncoder(BaseTextEncoder):
    """
    A mock text encoder for testing purposes.
    """
    def __init__(self):
        super().__init__()
        self.dim = 1024
        self.seq_len = 256

    def encode(self, text: list[str]) -> Tensor:
        """
        Encode the batch of text into a tensor.

        Args:
            text: The batch of text to encode. [B]

        Returns:
            The encoded tensor. [B, seq_len, dim]
        """
        embeddings = []
        for t in text:
            embeddings.append(torch.randn(self.seq_len, self.dim))
        return torch.stack(embeddings)