"""检查双曲分组在重合点、混合长度和全 PAD 级联上的行为。"""
from types import SimpleNamespace
import unittest

import torch

from module import GeometricKMeansLayer
from utils.Lorentz import exp_map, lorentz_sqdist


class GroupingTest(unittest.TestCase):
    def test_geodesic_distance(self):
        tangent = torch.tensor([[[0.], [2.], [-1.]]], dtype=torch.float64, requires_grad=True)
        points = exp_map(tangent)
        distances = lorentz_sqdist(points, points)
        expected = (tangent - tangent.transpose(1, 2)).square()
        torch.testing.assert_close(distances, expected)
        distances.sum().backward()
        self.assertTrue(torch.isfinite(tangent.grad).all())

    def test_padding_and_gradient(self):
        for space in ('hyperbolic', 'euclidean_matched'):
            args = SimpleNamespace(grouping_space=space, dim=3, group_num=3, deterministic_eval=True)
            layer = GeometricKMeansLayer(args).eval()
            x = torch.randn(2, 4, 3, requires_grad=True)
            mask = torch.tensor([[False, False, True, True], [True] * 4])
            groups, active, counts = layer(x, mask)
            alone = layer(x[:1, :2], mask[:1, :2])
            torch.testing.assert_close(groups[:1], alone[0])
            torch.testing.assert_close(counts.sum(dim=1), torch.tensor([2., 0.]))
            self.assertFalse(active[1].any() or active[:, -1].any())
            groups.square().sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad[~mask].abs().sum().item(), 0)
            self.assertEqual(x.grad[mask].abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(layer.cluster_centers.grad).all())
