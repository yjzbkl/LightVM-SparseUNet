import argparse
import json
from pathlib import Path

from models import LightVMSparseUNet
from utils.complexity import (
    append_optimization_log,
    profile_macs,
    try_fvcore_profile,
    try_ptflops_profile,
    try_thop_profile,
)
from utils.config import load_config


def build_model_from_config(config_path: str) -> LightVMSparseUNet:
    cfg = load_config(config_path)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    return LightVMSparseUNet(**model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mmotu.yaml")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-change", default="")
    args = parser.parse_args()

    model = build_model_from_config(args.config)
    input_size = (1, 3, args.height, args.width)
    result = profile_macs(model, input_size=input_size, device=args.device)
    result.thop_macs = try_thop_profile(model, input_size=input_size, device=args.device)
    result.fvcore_flops = try_fvcore_profile(model, input_size=input_size, device=args.device)
    result.ptflops_macs = try_ptflops_profile(model, input_size=input_size, device=args.device)

    passed = 0.080e6 <= result.total_params <= 0.085e6 and 0.155 <= result.gmacs <= 0.165
    payload = {
        "Total Params": result.total_params,
        "Trainable Params": result.trainable_params,
        "MACs": result.macs,
        "FLOPs": result.flops,
        "Paper GFLOPs MACs": result.gmacs,
        "Arithmetic GFLOPs 1MAC=2FLOPs": result.gflops_arithmetic,
        "Profiler Name": result.profiler_name,
        "Profiler Version": result.profiler_version,
        "Input Size": result.input_size,
        "THOP MACs": result.thop_macs,
        "FVCore FLOPs": result.fvcore_flops,
        "ptflops MACs": result.ptflops_macs,
        "Budget Passed": passed,
        "FLOPs Note": "The paper budget is compared against MACs/G, matching common THOP-style GFLOPs reporting. Arithmetic FLOPs are also reported as 1 MAC ~= 2 FLOPs.",
    }
    print(json.dumps(payload, indent=2))

    if args.log_change:
        append_optimization_log(
            "outputs/profile/complexity_optimization_log.csv",
            args.log_change,
            result.total_params,
            result.gmacs,
            passed,
        )


if __name__ == "__main__":
    main()
