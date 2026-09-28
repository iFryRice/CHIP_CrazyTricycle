"""Boundary-ranking contracts with synthetic spans and no model downloads."""

import importlib.util
import unittest

from patientphex.span_losses import boundary_ranking_loss


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch training dependency is not installed")
class BoundaryRankingLossTests(unittest.TestCase):
    def tensors(self, length=4, width=4):
        import torch

        logits = torch.zeros(1, length, width, 2, requires_grad=True)
        targets = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for start in range(length):
            mask[:, start, :min(width, length - start), :] = True
        return logits, targets, mask

    def test_wrong_endpoint_gets_gradient_but_unrelated_high_score_does_not(self):
        import torch

        logits, targets, mask = self.tensors()
        targets[0, 1, 1, 0] = 1
        with torch.no_grad():
            logits[0, 1, 1, 0] = 0.2
            logits[0, 1, 2, 0] = 3.0
            logits[0, 2, 0, 0] = 1.0
            logits[0, 0, 0, 0] = 9.0
        loss = boundary_ranking_loss(logits, targets, mask)
        self.assertAlmostEqual(loss.item(), 3.8, places=5)
        loss.backward()
        self.assertEqual(logits.grad[0, 1, 1, 0].item(), -1)
        self.assertEqual(logits.grad[0, 1, 2, 0].item(), 1)
        self.assertEqual(logits.grad[0, 2, 0, 0].item(), 0)
        self.assertEqual(logits.grad[0, 0, 0, 0].item(), 0)

    def test_same_end_negative_is_selected(self):
        import torch

        logits, targets, mask = self.tensors()
        targets[0, 1, 1, 0] = 1
        with torch.no_grad():
            logits[0, 0, 2, 0] = 4
            logits[0, 1, 2, 0] = 2
        loss = boundary_ranking_loss(logits, targets, mask)
        self.assertEqual(loss.item(), 5)
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 2, 0].item(), 1)
        self.assertEqual(logits.grad[0, 1, 2, 0].item(), 0)

    def test_nested_and_other_label_gold_spans_are_never_boundary_negatives(self):
        import torch

        logits, targets, mask = self.tensors()
        targets[0, 1, 0, 0] = 1
        targets[0, 1, 2, 0] = 1
        targets[0, 1, 1, 1] = 1
        with torch.no_grad():
            logits[0, 1, 0, 0] = 3
            logits[0, 1, 2, 0] = 10
            logits[0, 1, 1, :] = 20
        loss = boundary_ranking_loss(logits, targets, mask)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_ignored_candidate_in_any_channel_is_excluded(self):
        import torch

        logits, targets, mask = self.tensors()
        targets[0, 1, 0, 0] = 1
        mask[0, 1, 2, 1] = False
        with torch.no_grad():
            logits[0, 1, 0, 0] = 0.2
            logits[0, 1, 2, 0] = 5
            logits[0, 1, 1, 0] = 1
        loss = boundary_ranking_loss(logits, targets, mask)
        self.assertAlmostEqual(loss.item(), 1.8, places=5)
        loss.backward()
        self.assertEqual(logits.grad[0, 1, 2, 0].item(), 0)
        self.assertEqual(logits.grad[0, 1, 1, 0].item(), 1)

    def test_no_pairs_and_no_gold_return_differentiable_zero(self):
        for positive in (False, True):
            logits, targets, mask = self.tensors(length=1, width=3)
            if positive:
                targets[0, 0, 0, 0] = 1
            loss = boundary_ranking_loss(logits, targets, mask)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_malformed_shapes_and_negative_margin_are_rejected(self):
        logits, targets, mask = self.tensors()
        with self.assertRaises(ValueError):
            boundary_ranking_loss(logits, targets[:, :, :, :1], mask)
        for margin in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                boundary_ranking_loss(logits, targets, mask, margin)


if __name__ == "__main__":
    unittest.main()
