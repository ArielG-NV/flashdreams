import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb
from torch import Tensor
from torch.distributed import ProcessGroup

from flashsim.attention import BlockKVCache, RingAttention

@dataclass
class KVCache:
    k: Tensor  # [batch_size, seq_len, n_heads, head_dim]
    v: Tensor  # [batch_size, seq_len, n_heads, head_dim]
    _end: int = 0  # until where the kv cache is valid

    def __post_init__(self):
        assert self.k.ndim == 4, "k is expected to be 4D tensor with shape [batch_size, seq_len, n_heads, head_dim]"
        assert self.v.ndim == 4, "v is expected to be 4D tensor with shape [batch_size, seq_len, n_heads, head_dim]"
        assert self.k.shape == self.v.shape, "k and v must have the same shape"

    def update(self, k: Tensor, v: Tensor, start: int) -> None:
        assert k.ndim == 4, "k is expected to be 4D tensor with shape [batch_size, s, n_heads, head_dim]"
        assert v.ndim == 4, "v is expected to be 4D tensor with shape [batch_size, s, n_heads, head_dim]"
        end = start + k.shape[1]
        # print(f"[update kv cache] adding k, v to ({start}, {end})")

        if end > self.k.shape[1]:
            # extend the kv cache
            assert start == self.k.shape[1], "start must be equal to the current seq_len"
            self.k = torch.cat([self.k, k], dim=1)
            self.v = torch.cat([self.v, v], dim=1)
        else:
            # inplace update the kv cache
            self.k[:, start:end] = k
            self.v[:, start:end] = v
        self._end = end

    def shrink(self, sink_size: int = 0, local_attn_size: int = -1) -> None:
        if local_attn_size == -1:
            # global attention, we keep all tokens in the kv cache.
            return
        if self._end == 0:
            # empty kv cache, there is nothing to shrink.
            return

        assert local_attn_size > 0, "local_attn_size must be greater than 0"
        # local attention, we keep maximum {sink_size + local_attn_size}
        # tokens in the kv cache.
        sink_start = 0
        sink_end = min(self._end, sink_size)

        local_start = max(self._end - local_attn_size, sink_end)
        local_end = self._end

        self.k = torch.cat([self.k[:, sink_start:sink_end], self.k[:, local_start:local_end]], dim=1)
        self.v = torch.cat([self.v[:, sink_start:sink_end], self.v[:, local_start:local_end]], dim=1)
        self._end = self.k.shape[1]

        # print(f"[shrink kv cache] take ({sink_start}, {sink_end}) and ({local_start}, {local_end}), new end is {self._end}")


@dataclass
class CrossAttnKVCache:
    text: KVCache
    img: Optional[KVCache] = None  # only used for I2V


@dataclass
class AttentionBlockKVCache:
    self_attn: KVCache
    cross_attn: CrossAttnKVCache


@dataclass
class CausalWanNetworkCache:
    block_kv_caches: List[AttentionBlockKVCache]

    def __getitem__(self, index: int) -> AttentionBlockKVCache:
        return self.block_kv_caches[index]


def rope_apply(x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool = True) -> Tensor:
    """
    Optimized version of rope_apply using flash_attention's rotary embedding implementation.
    This version processes the entire batch at once for efficiency.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads, head_dim]
        cos (Tensor): Cosine frequencies with shape [seq_len, head_dim // 2]
        sin (Tensor): Sine frequencies with shape [seq_len, head_dim // 2]

    Returns:
        Tensor: Rotary-embedded tensor with same shape and dtype as input
    """
    assert x.ndim == 4 and cos.ndim == 2 and sin.ndim == 2, "x, cos, and sin must be 4D, 2D, and 2D tensors"
    rotary_dtype = cos.dtype
    rotated = apply_rotary_emb(x.to(rotary_dtype), cos, sin, interleaved=interleaved, inplace=False)
    return rotated.to(x.dtype)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: Tensor) -> Tensor:
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op = RingAttention(qkv_format="bshd", backend="cudnn")

    def set_context_parallel_group(self, cp_group: ProcessGroup):
        self.attn_op.set_context_parallel_group(cp_group=cp_group)

    def prepare_cache(
        self,
        max_seqlen: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> KVCache:
        r"""
        Preallocate the KV cache for self-attention.
        """
        k_cache = torch.zeros(1, max_seqlen, self.num_heads, self.head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, max_seqlen, self.num_heads, self.head_dim, device=device, dtype=dtype)
        return KVCache(k=k_cache, v=v_cache)

    def forward(
        self,
        x: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        current_start: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        kv_cache: Optional[KVCache] = None,
        verbose: bool = False,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            rotary_cos (Tensor): Cosine frequencies with shape [seq_len, head_dim // 2]
            rotary_sin (Tensor): Sine frequencies with shape [seq_len, head_dim // 2]
            current_start (int): Start index of the current sequence
            kv_cache (Optional[KVCache]): KV cache with shape [batch_size, seq_len, n_heads, head_dim].
                If None, will not use kv cache. If given, it will be inplace updated.

        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        assert x.ndim == 3, "x is expected to be 3D tensor with shape [batch_size, seq_len, n_heads * head_dim]"
        assert rotary_cos.dtype == rotary_sin.dtype == torch.float32, (
            "rotary_cos and rotary_sin are expected to be float32"
        )

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        x = x.view(b, s, n * d)
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # rope_apply takes in freqs that already bake in kv cache offset
        roped_query = rope_apply(q, rotary_cos, rotary_sin)
        roped_key = rope_apply(k, rotary_cos, rotary_sin)

        if kv_cache is not None:
            if local_attn_size == -1:
                kv_cache.update(roped_key, v, current_start)
                cached_k = kv_cache.k[:, : current_start + s]
                cached_v = kv_cache.v[:, : current_start + s]
            else:
                current_start = kv_cache._end
                kv_cache.update(roped_key, v, current_start)
                kv_cache.shrink(sink_size=sink_size, local_attn_size=local_attn_size)
                cached_k = kv_cache.k
                cached_v = kv_cache.v
        else:
            cached_k = roped_key
            cached_v = v
        x = self.attn_op(roped_query, cached_k, cached_v)

        x = x.reshape(b, s, n * d)
        x = self.o(x)
        return x


class WanCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        i2v: bool = False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.i2v = i2v

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op = RingAttention(qkv_format="bshd", backend="cudnn")

        if i2v:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.attn_op_image = RingAttention(qkv_format="bshd", backend="cudnn")

    def prepare_cache(self, context_text: Tensor, context_img: Optional[Tensor] = None) -> CrossAttnKVCache:
        r"""
        Precompute the KV cache for cross-attention.
        """
        assert context_text.ndim == 3, (
            "context_text is expected to be 3D tensor with shape [batch_size, seq_len, n_heads * head_dim]"
        )
        b, s, n, d = *context_text.shape[:2], self.num_heads, self.head_dim
        k_cache = self.norm_k(self.k(context_text)).view(b, s, n, d)
        v_cache = self.v(context_text).view(b, s, n, d)
        kv_cache_text = KVCache(k=k_cache, v=v_cache)

        if self.i2v:
            assert context_img is not None, "context_img is expected to be provided for I2V cross-attention"
            assert context_img.ndim == 3, (
                "context_img is expected to be 3D tensor with shape [batch_size, seq_len, n_heads * head_dim]"
            )
            b, s, n, d = *context_img.shape[:2], self.num_heads, self.head_dim
            k_cache_img = self.norm_k_img(self.k_img(context_img)).view(b, s, n, d)
            v_cache_img = self.v_img(context_img).view(b, s, n, d)
            kv_cache_img = KVCache(k=k_cache_img, v=v_cache_img)
            return CrossAttnKVCache(text=kv_cache_text, img=kv_cache_img)
        else:
            return CrossAttnKVCache(text=kv_cache_text)

    def forward(self, x: Tensor, kv_cache: CrossAttnKVCache) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            kv_cache (CrossAttnKVCache): Cross-attention KV cache.

        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        assert x.ndim == 3, "x is expected to be 3D tensor with shape [batch_size, seq_len, n_heads * head_dim]"

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        x = self.attn_op(q, kv_cache.text.k, kv_cache.text.v).view(b, s, n * d)
        if self.i2v:
            x_img = self.attn_op_image(q, kv_cache.img.k, kv_cache.img.v).view(b, s, n * d)
            x = x + x_img
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        i2v: bool = False,
        use_camera_cond: bool = False,
        cam_dim: int = 1536,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.i2v = i2v
        self.use_camera_cond = use_camera_cond
        self.cam_dim = cam_dim

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            eps=eps,
            i2v=i2v,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        if self.use_camera_cond:
            self.cam_encoder = nn.Linear(self.cam_dim, dim, bias=False)

        self.is_e_fused = False

    def prepare_cache(
        self, max_seqlen: int, context_text: Tensor, context_img: Optional[Tensor] = None
    ) -> AttentionBlockKVCache:
        r"""
        Prepare the KV cache for both self-attention and cross-attention.
        """
        device = context_text.device
        dtype = context_text.dtype
        cross_attn_kv_cache = self.cross_attn.prepare_cache(context_text, context_img)
        self_attn_kv_cache = self.self_attn.prepare_cache(max_seqlen, device=device, dtype=dtype)
        return AttentionBlockKVCache(self_attn=self_attn_kv_cache, cross_attn=cross_attn_kv_cache)

    def set_context_parallel_group(self, cp_group: ProcessGroup):
        self.self_attn.set_context_parallel_group(cp_group)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        block_kv_cache: AttentionBlockKVCache,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        current_start: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        camera: Optional[Tensor] = None,
        verbose: bool = False,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            e (Tensor): Modulation tensor with shape [batch_size, 6, n_heads * head_dim]
            block_kv_cache (AttentionBlockKVCache): KV cache for the attention block
            rotary_cos (Tensor): Cosine frequencies with shape [seq_len, head_dim // 2]
            rotary_sin (Tensor): Sine frequencies with shape [seq_len, head_dim // 2]
            current_start (int): Start index of the current sequence
            camera (Optional[Tensor]): Camera condition tensor with shape [batch_size, seq_len, cam_dim]
        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        e = (self.modulation + e).chunk(6, dim=1)  # [B, 1, D] each

        y = self.norm1(x) * (1 + e[1]) + e[0]  # [B, L, D]
        if self.use_camera_cond:
            assert camera is not None, "camera is expected to be provided for camera condition"
            cam_emb = self.cam_encoder(camera)
            y = y + cam_emb
        y = self.self_attn(
            y,
            rotary_cos,
            rotary_sin,
            current_start,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            kv_cache=block_kv_cache.self_attn,
            verbose=verbose,
        )
        x = x + (y * e[2])  # [B, L, D]

        x = x + self.cross_attn(self.norm3(x), kv_cache=block_kv_cache.cross_attn)
        y = self.norm2(x) * (1 + e[4]) + e[3]  # [B, L, D]
        y = self.ffn(y)
        x = x + (y * e[5])  # [B, L, D]
        return x

    def forward_with_e_fused(
        self,
        x: Tensor,
        block_kv_cache: AttentionBlockKVCache,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        current_start: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        camera: Optional[Tensor] = None,
        verbose: bool = False,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            block_kv_cache (AttentionBlockKVCache): KV cache for the attention block
            rotary_cos (Tensor): Cosine frequencies with shape [seq_len, head_dim // 2]
            rotary_sin (Tensor): Sine frequencies with shape [seq_len, head_dim // 2]
            current_start (int): Start index of the current sequence
            camera (Optional[Tensor]): Camera condition tensor with shape [batch_size, seq_len, cam_dim]

        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        assert self.is_e_fused, "Need to fuse input e into weights first"
        y = self.norm1(x)
        if self.use_camera_cond:
            assert camera is not None, "camera is expected to be provided for camera condition"
            cam_emb = self.cam_encoder(camera)
            y = y + cam_emb
        x = x + self.self_attn(
            y,
            rotary_cos,
            rotary_sin,
            current_start,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            kv_cache=block_kv_cache.self_attn,
            verbose=verbose,
        )
        y = self.norm3(x)
        x = x + self.cross_attn(y, kv_cache=block_kv_cache.cross_attn)
        y = self.norm2(x)
        x = x + self.ffn(y)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            e (Tensor): Modulation tensor with shape [batch_size, 1, n_heads * head_dim]

        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        assert x.ndim == 3, "x is expected to be 3D tensor with shape [batch_size, seq_len, n_heads * head_dim]"
        assert e.ndim == 3, "e is expected to be 3D tensor with shape [batch_size, 1, n_heads * head_dim]"

        # TODO(ruilong): These can be fused into a normlinear layer.
        e = (self.modulation + e).chunk(2, dim=1)  # [B, 1, D] each
        x = self.norm(x) * (1 + e[1]) + e[0]  # [B, L, D]
        x = self.head(x)
        return x



class CausalWanNetwork(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        concat_padding_mask: bool = False,
        use_camera_cond: bool = False,
        cam_dim: int = 1536,
        additional_concat_ch: int = 0,  # hdmap
        **kwargs,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            concat_padding_mask (`bool`, *optional*, defaults to False):
                Enable concat padding mask
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.concat_padding_mask = concat_padding_mask
        self.additional_concat_ch = additional_concat_ch

        # embeddings
        in_dim = in_dim + 1 if self.concat_padding_mask else in_dim
        self.patch_embedding = nn.Linear(in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)
        if additional_concat_ch > 0:
            self.additional_patch_embedding = nn.Linear(
                additional_concat_ch * patch_size[0] * patch_size[1] * patch_size[2], dim
            )

        # blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    i2v=(model_type == "i2v"),
                    use_camera_cond=use_camera_cond,
                    cam_dim=cam_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        self.cp_size = None
        self.is_shuffle_op_fused = False
        self.is_t_fused = False

    def initialize_context_parallel(self, cp_group: Optional[ProcessGroup] = None):
        """
        Set the context parallel group for the network.

        Must be called before preparing cache.
        """
        if cp_group is None:
            self.cp_size = 1
        else:
            self.cp_size = cp_group.size()
            for block in self.blocks:
                block.set_context_parallel_group(cp_group)

    def prepare_cache(
        self,
        H: int,
        W: int,
        prealloc_T: int,
        text_crossattn_emb: Tensor,  # umt5 text embedding. shape [1, 512, 4096]
        img_crossattn_emb: Optional[Tensor] = None,  # CLIP image embedding for I2V. shape [1, 256, 1280]
    ) -> CausalWanNetworkCache:
        """
        Prepare the cache for the network.

        Should be called after initializing the context parallel group.
        """
        assert self.cp_size is not None, (
            "Need to call initialize_context_parallel(Optional[ProcessGroup]) before preparing cache"
        )
        if self.model_type == "i2v":
            assert img_crossattn_emb is not None, "img_crossattn_emb is expected to be provided for I2V"

        patch_H = H // self.patch_size[1]
        patch_W = W // self.patch_size[2]
        prealloc_seqlen = prealloc_T * patch_H * patch_W
        assert prealloc_seqlen % self.cp_size == 0, (
            f"prealloc_seqlen {prealloc_seqlen} must be divisible by cp_size {self.cp_size}"
        )

        # prepare the cache for the text embedding, and optionally the image embedding
        context_text = self.text_embedding(text_crossattn_emb)  # [B, L1, D]
        if self.model_type == "i2v":
            context_img = self.img_emb(img_crossattn_emb)  # [B, L2, D]
        else:
            context_img = None

        # prepare the cache for the blocks
        block_kv_caches = []
        for block in self.blocks:
            block_kv_cache = block.prepare_cache(prealloc_seqlen // self.cp_size, context_text, context_img)
            block_kv_caches.append(block_kv_cache)

        torch.cuda.empty_cache()
        return CausalWanNetworkCache(block_kv_caches=block_kv_caches)

    def fuse_ops_into_weights(self, timesteps: Optional[Tensor] = None):
        """
        Fuse some ops into the weights of the network.

        Note this function should be called only after loading the checkpoint.

        Args:
            timesteps (Optional[Tensor]): Timesteps with shape [1] bfloat16 type
        """
        self._fuse_shuffle_op_into_head()

    def _fuse_shuffle_op_into_head(self):
        """
        In the WAN model, the patchify operation is
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",

        while the unpatchify operation is
        "b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)"

        This is likely a bug in the WAN model where the last dimension is shuffled after the network.

        To fix this, we could fuse this shuffle op into the last linear layer of the head,
        so that we do not have to do this shuffle op explicitly before returning the result.

        Calling this function to modify the head in place, is equivalent to the following code
        before returning the result:
        ```python
        x = rearrange(
            x,
            "B L (nt nh nw d) -> B L (d nt nh nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            d=self.out_dim,
        ) # [B, L, D]
        ```
        """
        if self.is_shuffle_op_fused:
            return

        self.head.head.weight.data = rearrange(
            self.head.head.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
            c=self.out_dim,
        ).contiguous()
        if self.head.head.bias is not None:
            self.head.head.bias.data = rearrange(
                self.head.head.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.patch_size[0],
                kh=self.patch_size[1],
                kw=self.patch_size[2],
                c=self.out_dim,
            ).contiguous()

        self.is_shuffle_op_fused = True

    def forward(
        self,
        x: Tensor,
        timesteps: Optional[Tensor],
        block_kv_caches: List[AttentionBlockKVCache],
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        current_start: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        camera: Optional[Tensor] = None,
        hdmap: Optional[Tensor] = None,
    ):
        r"""
        Args:
            x (Tensor): Input tensor with shape [B, L, D] after CP. The layout is assumed to be
                "b (t h w) (d nt nh nw)".
            timesteps (Optional[Tensor]): Timesteps with shape [B].
            block_kv_caches (List[AttentionBlockKVCache]): KV caches for the blocks.
            rotary_cos (Tensor): Cosine frequencies with shape [L, D] after CP.
            rotary_sin (Tensor): Sine frequencies with shape [L, D] after CP.
            current_start (int): Start token index of the current chunk
            camera (Optional[Tensor]): Camera condition tensor with shape [B, L, cam_dim] after CP.
                assuming same layout as x.
            hdmap (Optional[Tensor]): HDMap condition tensor with shape [B, L, additional_concat_ch] after CP.
                assuming same layout as x.
        """
        assert x.ndim == 3, "x is expected to be 3D tensor with shape [B, L, D]"
        assert rotary_cos.ndim == 2, "rotary_cos is expected to be 2D tensor with shape [L, D]"
        assert rotary_sin.ndim == 2, "rotary_sin is expected to be 2D tensor with shape [L, D]"
        assert rotary_cos.shape == rotary_sin.shape
        assert timesteps.ndim == 1, "timesteps is expected to be 2D tensor with shape [B]"
        assert self.is_shuffle_op_fused, "needs to call _fuse_shuffle_op_into_head() before running forward"

        # patch embedding
        x = self.patch_embedding(x)  # (B, L, D)

        # patch embedding for hdmap
        if self.additional_concat_ch > 0:
            assert hdmap is not None, "hdmap is expected to be provided for additional concat channels"
            additional_x = self.additional_patch_embedding(hdmap)
            x = x + additional_x  # (B, L, D)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x))  # [B, D]
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, D]

        # transformer blocks
        for block_idx, block in enumerate(self.blocks):
            x = block(
                x=x,
                e=e0,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                current_start=current_start,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                block_kv_cache=block_kv_caches[block_idx],
                camera=camera,
            )

        # head
        x = self.head(x, e.unsqueeze(1))  # (B, L, D)
        return x


def test_basic(i2v: bool = False, use_camera_cond: bool = False, use_hdmap: bool = False):
    import time
    
    torch.manual_seed(42)
    # 14B model
    device = "cuda"
    dtype = torch.bfloat16

    additional_concat_ch = 0
    if i2v:
        model_type = "i2v"
        in_dim = 16 + 20  # 16 is noise, 20 is image conditioning
        if use_hdmap:
            additional_concat_ch = 16
    else:
        model_type = "t2v"
        in_dim = 16

    T, H, W = 3, 720 // 8, 1280 // 8
    prealloc_T = 21
    num_tokens_per_frame = H // 2 * W // 2
    num_tokens_per_chunk = T * num_tokens_per_frame

    network = CausalWanNetwork(
        model_type=model_type,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        in_dim=in_dim,
        num_heads=40,
        num_layers=40,
        out_dim=16,
        text_len=512,
        use_camera_cond=use_camera_cond,
        cam_dim=1536,
        additional_concat_ch=additional_concat_ch,
    ).to(device=device, dtype=dtype)
    # torch.save(network.state_dict(), "outputs/wan2_1_network.pth")
    torch.load_state_dict(torch.load("outputs/wan2_1_network.pth"))

    torch.manual_seed(42)
    data = torch.randn(1, in_dim, T, H, W, device=device, dtype=dtype)
    x = rearrange(
        data,
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
        kt=network.patch_size[0],
        kh=network.patch_size[1],
        kw=network.patch_size[2],
    )
    timesteps = torch.randn(1, device=device, dtype=dtype)
    rotary_cos = torch.randn(num_tokens_per_chunk, 64, device=device, dtype=torch.float32)
    rotary_sin = torch.randn(num_tokens_per_chunk, 64, device=device, dtype=torch.float32)
    camera = torch.randn(1, num_tokens_per_chunk, 1536, device=device, dtype=dtype)
    if use_hdmap:
        hdmap = torch.randn(1, additional_concat_ch, T, H, W, device=device, dtype=dtype)
        hdmap = rearrange(
            hdmap,
            "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
            kt=network.patch_size[0],
            kh=network.patch_size[1],
            kw=network.patch_size[2],
        )

    else:
        hdmap = None

    network.initialize_context_parallel()
    network.fuse_ops_into_weights(timesteps)

    network_cache = network.prepare_cache(
        H=H,
        W=W,
        prealloc_T=prealloc_T,
        text_crossattn_emb=torch.randn(1, 512, 4096, device=device, dtype=dtype),
        img_crossattn_emb=torch.randn(1, 256, 1280, device=device, dtype=dtype),
    )

    @torch.no_grad()
    def _run():
        output = network(
            x,
            timesteps,
            network_cache,
            rotary_cos,
            rotary_sin,
            current_start=0,
            local_attn_size=21 * num_tokens_per_frame,
            sink_size=3 * num_tokens_per_frame,
            camera=camera,
            hdmap=hdmap,
        )
        return output

    output = _run()  # warmup

    iterations = 10
    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(iterations):
        _run()
    torch.cuda.synchronize()
    toc = time.time()
    print(f"Elapsed time per run: {(toc - tic) / iterations} seconds")
    # Elapsed time per run: 0.6835304498672485 seconds

    print(
        "i2v:",
        i2v,
        "use_camera_cond:",
        use_camera_cond,
        "use_hdmap:",
        use_hdmap,
        "x.shape:",
        x.shape,
        "output.shape:",
        output.shape,
        "output.sum():",
        output.sum(),
    )
    # Elapsed time per run: 0.6563318490982055 seconds
    # i2v: True use_camera_cond: False use_hdmap: False x.shape: torch.Size([1, 10800, 144]) output.shape: torch.Size([1, 10800, 64]) output.sum(): tensor(10304., device='cuda:0', dtype=torch.bfloat16)


# torchrun --nproc_per_node=1 flashsim/model/video_dit/wan2_1/network.py
if __name__ == "__main__":
    test_basic(i2v=True)
    # test_basic(i2v=False)
    # test_basic(i2v=True, use_camera_cond=True)
    # test_basic(i2v=True, use_hdmap=True)
