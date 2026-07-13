import tempfile
import unittest
from pathlib import Path

import torch

from losses import BCEDiceLossWithLogits
from metrics import binary_segmentation_metrics
from models import LightVMSparseUNet
from models.modules import MultiPathVisualMamba, SparseSamplingSelfAttention
from utils import load_checkpoint, save_checkpoint


class ModuleAndTrainingTests(unittest.TestCase):
    def test_paper_metrics_iou_miou_hd_acc(self):
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, 2:6, 2:6] = 1
        logits = torch.where(target.bool(), torch.tensor(20.0), torch.tensor(-20.0))
        metrics = binary_segmentation_metrics(logits, target)
        for name in ("iou", "miou", "hd", "acc"):
            self.assertIn(name, metrics)
        self.assertAlmostEqual(metrics["iou"], 1.0)
        self.assertAlmostEqual(metrics["miou"], 1.0)
        self.assertAlmostEqual(metrics["hd"], 0.0)
        self.assertAlmostEqual(metrics["acc"], 1.0)

    def test_hausdorff_empty_mask_is_finite(self):
        logits = torch.full((1, 1, 6, 8), -20.0)
        target = torch.zeros_like(logits)
        target[:, :, 2, 3] = 1
        metrics = binary_segmentation_metrics(logits, target)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["hd"])))
        self.assertAlmostEqual(metrics["hd"], 10.0)

    def test_miou_is_foreground_background_arithmetic_mean(self):
        # TP=1, TN=1, FP=1, FN=1, so foreground IoU=background IoU=1/3.
        target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        predicted_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        logits = torch.where(predicted_mask.bool(), torch.tensor(20.0), torch.tensor(-20.0))
        metrics = binary_segmentation_metrics(logits, target, compute_hd=False)
        expected = 0.5 * (1 / 3 + 1 / 3)
        self.assertAlmostEqual(metrics["miou"], expected)

    def test_mvm_branches_forward_backward(self):
        for branches in (1, 2, 4, 8):
            module = MultiPathVisualMamba(
                channels=16,
                num_branches=branches,
                d_state=3,
                d_conv=3,
                expand=0.3125,
                projection_group_size=4,
            )
            x = torch.randn(2, 16, 8, 8, requires_grad=True)
            y = module(x)
            self.assertEqual(tuple(y.shape), tuple(x.shape))
            loss = y.square().mean()
            loss.backward()
            self.assertIsNotNone(x.grad)

    def test_sssa_sparse_sampling_padding_restore(self):
        sssa = SparseSamplingSelfAttention(dim=1, sparse_rate=3, num_heads=1)
        x = torch.arange(1 * 1 * 5 * 7, dtype=torch.float32).view(1, 1, 5, 7)
        x_pad, _, _, _ = sssa._pad(x)
        domains, _, h_tiles, w_tiles, _ = sssa._to_sparse_domains(x_pad)
        self.assertEqual(h_tiles, 2)
        self.assertEqual(w_tiles, 3)
        offset_i, offset_j = 1, 2
        domain_index = offset_i * sssa.sparse_rate + offset_j
        self.assertEqual(float(domains[domain_index, 0, 0]), float(x_pad[0, 0, 1, 2]))
        restored = sssa._from_sparse_domains(domains, 1, h_tiles, w_tiles, 1)
        self.assertTrue(torch.equal(restored[:, :, :5, :7], x))
        y = sssa(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_sssa_shared_parameters(self):
        model = LightVMSparseUNet()
        core_ids = [id(module.core) for module in model.skip_attentions]
        self.assertEqual(len(set(core_ids)), 1)
        first_q = model.skip_attentions[0].core.q_proj.weight
        last_q = model.skip_attentions[-1].core.q_proj.weight
        self.assertIs(first_q, last_q)

    def test_full_model_forward_standard_and_nonstandard(self):
        model = LightVMSparseUNet().eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 224, 224))
            self.assertEqual(tuple(y.shape), (1, 1, 224, 224))
            y2 = model(torch.randn(1, 3, 71, 83))
            self.assertEqual(tuple(y2.shape), (1, 1, 71, 83))

    def test_synthetic_training_step_and_checkpoint(self):
        model = LightVMSparseUNet()
        criterion = BCEDiceLossWithLogits()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        x = torch.randn(2, 3, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.pth"
            save_checkpoint(str(ckpt), model, optimizer=optimizer, epoch=1, metrics={"iou": 0.1})
            reloaded = LightVMSparseUNet()
            load_checkpoint(str(ckpt), reloaded)
            out = reloaded(torch.randn(1, 3, 64, 64))
            self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
