import unittest

from models import LightVMSparseUNet
from utils.complexity import profile_macs


class ComplexityBudgetTests(unittest.TestCase):
    def test_params_and_flops_budget(self):
        model = LightVMSparseUNet()
        result = profile_macs(model, input_size=(1, 3, 224, 224), device="cpu")
        total_params = result.total_params
        gflops = result.gmacs
        print(
            {
                "Total Params": total_params,
                "Trainable Params": result.trainable_params,
                "MACs": result.macs,
                "FLOPs": result.flops,
                "Profiler": result.profiler_name,
                "Profiler Version": result.profiler_version,
                "Input Size": result.input_size,
            }
        )
        self.assertGreaterEqual(total_params, 0.080e6)
        self.assertLessEqual(total_params, 0.085e6)
        self.assertGreaterEqual(gflops, 0.155)
        self.assertLessEqual(gflops, 0.165)


if __name__ == "__main__":
    unittest.main()
