from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .modules.multi_path_visual_mamba import MultiPathVisualMamba
from .modules.sparse_sampling_attention import (
    SharedSSSASkip,
    SparseSamplingSelfAttention,
)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depthwise_separable: bool = False,
    ) -> None:
        super().__init__()
        if depthwise_separable:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _group_norm(out_channels),
                nn.GELU(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                _group_norm(out_channels),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels % out_channels != 0:
            raise ValueError("in_channels must be divisible by out_channels")
        self.reduce = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=out_channels,
            bias=False,
        )
        if out_channels <= 64:
            pointwise_groups = 2
        else:
            pointwise_groups = max(1, out_channels // 4)
        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                groups=pointwise_groups,
                bias=False,
            ),
            _group_norm(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = self.refine(x)
        return self.act(x + skip)


class LightVMSparseUNet(nn.Module):
    """LightVM-SparseUNet with 6-level U-shaped encoder-decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: Sequence[int] = (16, 32, 64, 128, 256, 512),
        mvm_num_branches: int = 8,
        mvm_d_state: int = 5,
        mvm_d_conv: int = 3,
        mvm_expand: float = 0.3125,
        mvm_dt_rank: int = 1,
        mvm_projection_group_size: int = 8,
        mvm_backend: str = "selective_scan",
        sssa_enabled: bool = True,
        sssa_sparse_rate: int = 32,
        sssa_attention_dim: int = 1,
        sssa_num_heads: int = 1,
    ) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("LightVM-SparseUNet requires exactly 6 channel levels")
        self.channels = list(channels)
        self.sssa_enabled = sssa_enabled

        c = self.channels
        self.encoder1 = ConvNormAct(in_channels, c[0], depthwise_separable=False)
        self.encoder2 = ConvNormAct(c[0], c[1], depthwise_separable=False)
        self.encoder3 = ConvNormAct(c[1], c[2], depthwise_separable=True)
        self.encoder4 = nn.Sequential(
            MultiPathVisualMamba(
                c[3],
                num_branches=mvm_num_branches,
                d_state=mvm_d_state,
                d_conv=mvm_d_conv,
                expand=mvm_expand,
                dt_rank=mvm_dt_rank,
                projection_group_size=mvm_projection_group_size,
                backend=mvm_backend,
            ),
            nn.GELU(),
        )
        self.encoder5 = nn.Sequential(
            MultiPathVisualMamba(
                c[4],
                num_branches=mvm_num_branches,
                d_state=mvm_d_state,
                d_conv=mvm_d_conv,
                expand=mvm_expand,
                dt_rank=mvm_dt_rank,
                projection_group_size=mvm_projection_group_size,
                backend=mvm_backend,
            ),
            nn.GELU(),
        )
        self.encoder6 = nn.Sequential(
            MultiPathVisualMamba(
                c[5],
                num_branches=mvm_num_branches,
                d_state=mvm_d_state,
                d_conv=mvm_d_conv,
                expand=mvm_expand,
                dt_rank=mvm_dt_rank,
                projection_group_size=mvm_projection_group_size,
                backend=mvm_backend,
            ),
            nn.GELU(),
        )

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transition4 = nn.Conv2d(c[2], c[3], kernel_size=1, groups=c[2], bias=False)
        self.transition5 = nn.Conv2d(c[3], c[4], kernel_size=1, groups=c[3], bias=False)
        self.transition6 = nn.Conv2d(c[4], c[5], kernel_size=1, groups=c[4], bias=False)

        if sssa_enabled:
            self.shared_sssa_core = SparseSamplingSelfAttention(
                dim=sssa_attention_dim,
                sparse_rate=sssa_sparse_rate,
                num_heads=sssa_num_heads,
            )
            self.skip_attentions = nn.ModuleList(
                [
                    SharedSSSASkip(level_channels, sssa_attention_dim, self.shared_sssa_core)
                    for level_channels in c
                ]
            )
        else:
            self.shared_sssa_core = None
            self.skip_attentions = nn.ModuleList([nn.Identity() for _ in c])

        self.decoder5 = DecoderBlock(c[5], c[4])
        self.decoder4 = DecoderBlock(c[4], c[3])
        self.decoder3 = DecoderBlock(c[3], c[2])
        self.decoder2 = DecoderBlock(c[2], c[1])
        self.decoder1 = DecoderBlock(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.encoder1(x)
        f2 = self.encoder2(self.down(f1))
        f3 = self.encoder3(self.down(f2))
        f4_in = self.transition4(self.down(f3))
        f4 = self.encoder4(f4_in)
        f5_in = self.transition5(self.down(f4))
        f5 = self.encoder5(f5_in)
        f6_in = self.transition6(self.down(f5))
        f6 = self.encoder6(f6_in)
        return [f1, f2, f3, f4, f5, f6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self._encode(x)
        skips = [attn(feat) for attn, feat in zip(self.skip_attentions, features)]

        x = skips[5]
        x = self.decoder5(x, skips[4])
        x = self.decoder4(x, skips[3])
        x = self.decoder3(x, skips[2])
        x = self.decoder2(x, skips[1])
        x = self.decoder1(x, skips[0])
        logits = self.head(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits

    @classmethod
    def from_config(cls, cfg: Dict[str, object]) -> "LightVMSparseUNet":
        model_cfg = dict(cfg)
        model_cfg.pop("name", None)
        return cls(**model_cfg)


__all__ = ["LightVMSparseUNet"]
