import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SparseSamplingSelfAttention(nn.Module):
    """Sparse-Sampling Self-Attention core with interleaved spatial sampling.

    For sparse rate ``S``, the feature map is decomposed into ``S x S`` domains:
    ``x[:, :, i::S, j::S]``. Attention is computed independently inside each
    domain and then scattered back to the input positions.
    """

    def __init__(
        self,
        dim: int,
        sparse_rate: int = 32,
        num_heads: int = 1,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if sparse_rate <= 0:
            raise ValueError("sparse_rate must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.sparse_rate = sparse_rate
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        bsz, _, height, width = x.shape
        pad_h = (self.sparse_rate - height % self.sparse_rate) % self.sparse_rate
        pad_w = (self.sparse_rate - width % self.sparse_rate) % self.sparse_rate
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        mask = x.new_ones(bsz, 1, height, width, dtype=torch.bool)
        mask = F.pad(mask, (0, pad_w, 0, pad_h), value=False)
        return x_pad, mask, height, width

    def _to_sparse_domains(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        bsz, channels, height, width = x.shape
        s = self.sparse_rate
        h_tiles = height // s
        w_tiles = width // s
        domains = (
            x.contiguous()
            .view(bsz, channels, h_tiles, s, w_tiles, s)
            .permute(0, 3, 5, 2, 4, 1)
            .contiguous()
            .view(bsz * s * s, h_tiles * w_tiles, channels)
        )
        return domains, bsz, h_tiles, w_tiles, channels

    def _from_sparse_domains(
        self,
        domains: torch.Tensor,
        bsz: int,
        h_tiles: int,
        w_tiles: int,
        channels: int,
    ) -> torch.Tensor:
        s = self.sparse_rate
        return (
            domains.contiguous()
            .view(bsz, s, s, h_tiles, w_tiles, channels)
            .permute(0, 5, 3, 1, 4, 2)
            .contiguous()
            .view(bsz, channels, h_tiles * s, w_tiles * s)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        bsz, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {channels}")

        x_pad, valid_mask, orig_h, orig_w = self._pad(x)
        domains, bsz, h_tiles, w_tiles, channels = self._to_sparse_domains(x_pad)
        mask_domains, _, _, _, _ = self._to_sparse_domains(valid_mask.float())
        mask_domains = mask_domains.squeeze(-1).bool()

        residual = domains
        domains = self.norm(domains)
        q = self.q_proj(domains)
        k = self.k_proj(domains)
        v = self.v_proj(domains)

        total_domains, tokens, _ = q.shape
        q = q.view(total_domains, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(total_domains, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(total_domains, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~mask_domains[:, None, None, :], -1.0e4)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out * mask_domains[:, None, :, None].to(out.dtype)
        out = out.transpose(1, 2).contiguous().view(total_domains, tokens, channels)
        out = self.out_proj(out) + residual

        restored = self._from_sparse_domains(out, bsz, h_tiles, w_tiles, channels)
        return restored[:, :, :orig_h, :orig_w]


class SharedSSSASkip(nn.Module):
    """Per-level adapters around a shared SparseSamplingSelfAttention core."""

    def __init__(self, in_channels: int, attention_dim: int, core: SparseSamplingSelfAttention):
        super().__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.core = core
        self.in_adapter = nn.Conv2d(in_channels, attention_dim, kernel_size=1, bias=True)
        self.out_adapter = nn.Conv2d(attention_dim, in_channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_adapter(x)
        z = self.core(z)
        return x + self.gamma * self.out_adapter(z)


__all__ = ["SparseSamplingSelfAttention", "SharedSSSASkip"]
