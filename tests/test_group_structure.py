"""用同一条测地线上的已知距离检验两种分组约束。"""
import math
from types import SimpleNamespace
import unittest

import torch

from module import GeometricKMeansLayer
from utils.Lorentz import exp_map


class GroupStructureTest(unittest.TestCase):
    def layer(self, space, **kwargs):
        return GeometricKMeansLayer(SimpleNamespace(grouping_space=space, group_num=4, dim=1,
                                                    group_sep_weight=1, group_pref_weight=1,
                                                    group_max_iter=2, deterministic_eval=True, **kwargs))

    def test_loss_values_and_center_gradients(self):
        for space in ('euclidean_matched', 'hyperbolic'):
            layer = self.layer(space)
            # 用户间距 2，目标间距 1；第三个群无成员，第三个位置是 PAD。
            users = torch.tensor([[[-1.], [1.], [4.]]], dtype=torch.double, requires_grad=True)
            seeds = torch.tensor([[[-0.1], [0.1], [3.]]], dtype=torch.double, requires_grad=True)
            points = exp_map(users) if layer.hyperbolic else users
            centers = exp_map(seeds) if layer.hyperbolic else seeds
            sep, pref = layer.structure(points, centers, torch.ones(1, 3, dtype=torch.bool),
                                        torch.tensor([[False, False, True]]), 1, layer.squared_distances)
            self.assertAlmostEqual(sep.item(), 0.8**2, places=8)
            self.assertAlmostEqual(pref.item(), math.log(2) - 0.4, places=8)
            for loss in (sep, pref):
                grad, = torch.autograd.grad(loss, seeds, retain_graph=True)
                self.assertGreater(grad[0, 0].item(), 0)
                self.assertLess(grad[0, 1].item(), 0)
                self.assertEqual(grad[0, 2].item(), 0)
            (sep + pref).backward()
            self.assertEqual(users.grad[0, 2].item(), 0)

    def test_short_cascades_and_one_group(self):
        for space in ('euclidean_matched', 'hyperbolic'):
            layer = self.layer(space)
            inputs = torch.zeros(3, 2, 1, dtype=torch.double, requires_grad=True)
            points = exp_map(inputs) if layer.hyperbolic else inputs
            mask = torch.tensor([[False, False], [False, True], [True, True]])
            active = torch.tensor([[True, False], [True, True], [False, False]])
            sep, pref = layer.structure(points, points, active, mask, 1, layer.squared_distances)
            self.assertEqual((sep.item(), pref.item()), (0, 0))
            (sep + pref).backward()
            torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs))

    def test_structure_gradient_through_two_updates(self):
        for space in ('euclidean_matched', 'hyperbolic'):
            layer = self.layer(space).eval()
            with torch.no_grad():
                layer.cluster_centers[:, 0] = torch.tensor([-.1, .1, 0.])
            x = torch.tensor([[[-.5], [.5]]], requires_grad=True)
            (_, (sep, pref)) = layer(x, torch.zeros(1, 2, dtype=torch.bool), return_structure=True)
            for loss in (sep, pref):
                for grad in torch.autograd.grad(loss, (x, layer.cluster_centers), retain_graph=True):
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
