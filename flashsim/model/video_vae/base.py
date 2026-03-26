from abc import ABC, abstractmethod

from torch import Tensor


class BaseVideoVAE[EncoderCacheType, DecoderCacheType](ABC):

    @abstractmethod
    def initialize_encode_cache(self) -> EncoderCacheType:
        """
        Initialize the cache for encoding.
        """
        ...

    @abstractmethod
    def encode(self, x: Tensor, cache: EncoderCacheType) -> Tensor:
        """
        Encode a video into a latent representation.

        Args:
            x: The video to encode. [..., T, C, H, W]
            cache: The cache to use for encoding.

        Returns:
            The latent representation. [..., Tl, Cl, Hl, Wl]
        """
        ...

    @abstractmethod
    def initialize_decode_cache(self) -> DecoderCacheType:
        """
        Initialize the cache for decoding.
        """
        ...

    @abstractmethod
    def decode(self, z: Tensor, cache: DecoderCacheType) -> Tensor:
        """
        Decode a latent representation into a video.

        Args:
            z: The latent representation to decode. [..., Tl, Cl, Hl, Wl]
            cache: The cache to use for decoding.

        Returns:
            The video. [..., T, C, H, W]
        """
        ...
