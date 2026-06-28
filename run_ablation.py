import argparse
import copy

from models import LightVMSparseUNet
from utils.complexity import profile_macs
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mmotu.yaml")
    parser.add_argument("--branches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--sssa", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for branches in args.branches:
        for sssa in args.sssa:
            model_cfg = copy.deepcopy(cfg["model"])
            model_cfg["mvm_num_branches"] = branches
            model_cfg["sssa_enabled"] = sssa == "on"
            model = LightVMSparseUNet.from_config(model_cfg)
            result = profile_macs(model, device=args.device)
            print(
                f"branches={branches} sssa={sssa} params={result.total_params} "
                f"macs_g={result.gmacs:.6f}"
            )


if __name__ == "__main__":
    main()
