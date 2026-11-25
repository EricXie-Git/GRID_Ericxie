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

    def test_temperature_and_group_targets(self):
        for space in ('hyperbolic', 'euclidean_matched'):
            args = SimpleNamespace(grouping_space=space, dim=1, group_num=4)
            layer = GeometricKMeansLayer(args)
            users = torch.tensor([[1.], [-1.], [0.], [1.]], dtype=torch.float64)
            centers = torch.tensor([[[-1.], [1.], [1.]]], dtype=torch.float64).expand(4, -1, -1)
            if layer.hyperbolic:
                centers = exp_map(centers)
            active = torch.tensor([[True, True, False]] * 3 + [[False, False, False]])
            targets = layer.next_group_targets(users, centers, active, torch.tensor([1, 2, 3, 4]))
            self.assertEqual(targets.tolist(), [1, 0, -100, -100])
            point = exp_map(users[:1]) if layer.hyperbolic else users[:1]
            inputs = (point.unsqueeze(1), centers[:1, :2], torch.tensor([[False]]))
            warm = layer.assign(*inputs)
            layer.temperature = 0.1
            self.assertGreater(layer.assign(*inputs).max(), warm.max())

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
