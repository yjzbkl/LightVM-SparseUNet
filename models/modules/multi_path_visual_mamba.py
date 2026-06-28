import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class SelectiveScanMambaBlock(nn.Module):
    """Small Mamba-style selective state-space block for visual token streams.

    This implementation is kept as a pure PyTorch fallback so the project remains
    runnable when the CUDA extension is not available. It still performs
    input-dependent selective SSM recurrence and is not a convolution, MLP, or
    attention substitute.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        d_conv: int = 3,
        expand: float = 0.25,
        dt_rank: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.inner_dim = max(1, int(round(d_model * expand)))

        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, -2.0)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective scan.

        Args:
            x: Tensor with shape ``B x L x C``.
        Returns:
            Tensor with shape ``B x L x C``.
        """

        if x.dtype == torch.float16:
            x = x.float()
        bsz, length, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[..., :length].transpose(1, 2)
        x_conv = F.silu(x_conv)

        params = self.x_proj(x_conv)
        dt_raw, b_param, c_param = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))
        a = -torch.exp(self.A_log).to(dtype=x_conv.dtype)

        state = x_conv.new_zeros(bsz, self.inner_dim, self.d_state)
        outputs: List[torch.Tensor] = []
        for t in range(length):
            dt_t = dt[:, t, :]
            x_t = x_conv[:, t, :]
            b_t = b_param[:, t, :]
            c_t = c_param[:, t, :]

            d_a = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            d_b = dt_t.unsqueeze(-1) * b_t.unsqueeze(1)
            state = state * d_a + x_t.unsqueeze(-1) * d_b
            y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1) + self.D * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class ExternalMambaBlock(nn.Module):
    """Adapter around ``mamba_ssm.Mamba`` matching the local block interface."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        d_conv: int = 3,
        expand: float = 0.25,
        **_: object,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "mamba_ssm is not installed. Use mvm_backend=selective_scan."
            ) from exc
        self.block = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiPathVisualMamba(nn.Module):
    """Multi-path Visual Mamba module from LightVM-SparseUNet.

    ``X -> LN -> channel split -> independent Visual Mamba branches ->
    theta residual -> concat -> grouped channel projection -> LN``.
    """

    def __init__(
        self,
        channels: int,
        num_branches: int = 8,
        d_state: int = 4,
        d_conv: int = 3,
        expand: float = 0.25,
        dt_rank: int = 1,
        projection_group_size: int = 8,
        backend: str = "selective_scan",
    ) -> None:
        super().__init__()
        if channels % num_branches != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_branches ({num_branches})"
            )
        if num_branches not in (1, 2, 4, 8):
            raise ValueError("num_branches must be one of 1, 2, 4, 8")

        self.channels = channels
        self.num_branches = num_branches
        self.branch_dim = channels // num_branches
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.projection_group_size = projection_group_size
        self.backend = backend

        self.pre_norm = nn.LayerNorm(channels)
        block_cls = ExternalMambaBlock if backend == "mamba_ssm" else SelectiveScanMambaBlock
        self.branches = nn.ModuleList(
            [
                block_cls(
                    self.branch_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    bias=False,
                )
                for _ in range(num_branches)
            ]
        )
        self.theta = nn.Parameter(torch.ones(num_branches))

        if projection_group_size <= 0:
            projection_groups = 1
        else:
            projection_groups = max(1, channels // projection_group_size)
        while channels % projection_groups != 0:
            projection_groups -= 1
        self.channel_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            groups=projection_groups,
            bias=False,
        )
        self.projection_groups = projection_groups
        self.post_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        bsz, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")

        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.pre_norm(tokens)
        chunks = torch.chunk(tokens, self.num_branches, dim=-1)
        branch_outputs = []
        for idx, (chunk, branch) in enumerate(zip(chunks, self.branches)):
            branch_outputs.append(branch(chunk) + self.theta[idx] * chunk)

        merged = torch.cat(branch_outputs, dim=-1)
        merged_2d = merged.transpose(1, 2).reshape(bsz, channels, height, width)
        projected = self.channel_proj(merged_2d)
        projected_tokens = projected.flatten(2).transpose(1, 2)
        out = self.post_norm(projected_tokens)
        return out.transpose(1, 2).reshape(bsz, channels, height, width)


__all__ = ["MultiPathVisualMamba", "SelectiveScanMambaBlock"]
