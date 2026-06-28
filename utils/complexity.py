import csv
import copy
import os
import sys
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from models.modules.multi_path_visual_mamba import SelectiveScanMambaBlock
from models.modules.sparse_sampling_attention import SparseSamplingSelfAttention


@dataclass
class ProfileResult:
    total_params: int
    trainable_params: int
    macs: int
    flops: int
    profiler_name: str
    profiler_version: str
    input_size: Tuple[int, int, int, int]
    thop_macs: Optional[float] = None
    fvcore_flops: Optional[float] = None
    ptflops_macs: Optional[float] = None

    @property
    def gmacs(self) -> float:
        return self.macs / 1.0e9

    @property
    def gflops_arithmetic(self) -> float:
        return self.flops / 1.0e9


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    out = output
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
        module.in_channels // module.groups
    )
    return int(out.numel() * kernel_ops)


def _linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def _norm_macs(input_tensor: torch.Tensor) -> int:
    return int(input_tensor.numel() * 2)


def _selective_scan_extra_macs(module: SelectiveScanMambaBlock, x: torch.Tensor) -> int:
    bsz, length, _ = x.shape
    conv_macs = bsz * length * module.inner_dim * module.d_conv
    recurrence_macs = bsz * length * (4 * module.inner_dim * module.d_state + 2 * module.inner_dim)
    return int(conv_macs + recurrence_macs)


def _sssa_attention_macs(module: SparseSamplingSelfAttention, x: torch.Tensor) -> int:
    bsz, _, height, width = x.shape
    s = module.sparse_rate
    h_tiles = (height + s - 1) // s
    w_tiles = (width + s - 1) // s
    tokens = h_tiles * w_tiles
    domains = bsz * s * s
    return int(2 * domains * module.num_heads * tokens * tokens * module.head_dim)


def _decoder_upsample_macs(inputs) -> int:
    x, skip = inputs[0], inputs[1]
    bsz, channels, _, _ = x.shape
    height, width = skip.shape[-2:]
    return int(bsz * channels * height * width * 4)


def profile_macs(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    include_norm: bool = True,
    device: str = "cpu",
) -> ProfileResult:
    model = model.to(device)
    model.eval()
    macs = {"value": 0}
    handles = []

    def add(value: int) -> None:
        macs["value"] += int(value)

    def hook(module: nn.Module, inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            out = output[0]
        else:
            out = output
        if isinstance(module, nn.Conv2d):
            add(_conv2d_macs(module, out))
        elif isinstance(module, nn.Linear):
            add(_linear_macs(module, out))
        elif include_norm and isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            add(_norm_macs(inputs[0]))
        elif isinstance(module, SelectiveScanMambaBlock):
            add(_selective_scan_extra_macs(module, inputs[0]))
        elif isinstance(module, SparseSamplingSelfAttention):
            add(_sssa_attention_macs(module, inputs[0]))
        elif module.__class__.__name__ == "DecoderBlock":
            add(_decoder_upsample_macs(inputs))

    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.Linear,
                nn.LayerNorm,
                nn.GroupNorm,
                SelectiveScanMambaBlock,
                SparseSamplingSelfAttention,
            ),
        ) or module.__class__.__name__ == "DecoderBlock":
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        dummy = torch.randn(*input_size, device=device)
        model(dummy)

    for handle in handles:
        handle.remove()

    total_params, trainable_params = count_parameters(model)
    return ProfileResult(
        total_params=total_params,
        trainable_params=trainable_params,
        macs=macs["value"],
        flops=macs["value"] * 2,
        profiler_name="LightVM analytical MAC profiler",
        profiler_version="1.0",
        input_size=input_size,
    )


def try_thop_profile(model: nn.Module, input_size=(1, 3, 224, 224), device: str = "cpu") -> Optional[float]:
    extra_paths = os.environ.get("LIGHTVM_EXTRA_SITE_PACKAGES", "")
    for extra_path in [p for p in extra_paths.split(os.pathsep) if p]:
        if extra_path not in sys.path:
            sys.path.append(extra_path)
    try:
        from thop import profile
    except Exception:
        return None
    model = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(macs)


def try_fvcore_profile(model: nn.Module, input_size=(1, 3, 224, 224), device: str = "cpu") -> Optional[float]:
    extra_paths = os.environ.get("LIGHTVM_EXTRA_SITE_PACKAGES", "")
    for extra_path in [p for p in extra_paths.split(os.pathsep) if p]:
        if extra_path not in sys.path:
            sys.path.append(extra_path)
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    model = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        analysis = FlopCountAnalysis(model, dummy)
        return float(analysis.total())


def try_ptflops_profile(model: nn.Module, input_size=(1, 3, 224, 224), device: str = "cpu") -> Optional[float]:
    extra_paths = os.environ.get("LIGHTVM_EXTRA_SITE_PACKAGES", "")
    for extra_path in [p for p in extra_paths.split(os.pathsep) if p]:
        if extra_path not in sys.path:
            sys.path.append(extra_path)
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None
    model = copy.deepcopy(model).to(device).eval()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            macs, _ = get_model_complexity_info(
                model,
                input_size[1:],
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
                backend="pytorch",
            )
        except TypeError:
            macs, _ = get_model_complexity_info(
                model,
                input_size[1:],
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
    if macs is None:
        return None
    return float(macs)


def append_optimization_log(
    path: str,
    change: str,
    params: int,
    gmacs: float,
    passed: bool,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    exists = path_obj.exists()
    with path_obj.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["change", "Params", "FLOPs_G_paper_MACs", "passed"])
        writer.writerow([change, params, f"{gmacs:.6f}", "yes" if passed else "no"])
