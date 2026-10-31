"""双曲图编码的几何、梯度和边界检查，只使用合成小图。"""

import copy
import tempfile
from pathlib import Path
import unittest

import torch
from torch_geometric.data import Data

from config import parse_args, setup_info
from model import GRID
from module import LorentzGNNConv, LorentzGNNEncoder
from utils.Lorentz import exp_map, log_map, lorentz_centroid, MAX_TANGENT_NORM
from utils.Checkpoint import snapshot_state, save_checkpoint, load_checkpoint
import test_grid


class LorentzTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)

    def assert_on_manifold(self, x):
        norm = x[..., 0].square() - x[..., 1:].square().sum(dim=-1)
        torch.testing.assert_close(norm, torch.ones_like(norm), rtol=1e-9, atol=1e-9)
        self.assertTrue((x[..., 0] >= 1).all())

    def test_maps_and_gradients(self):
        tangent = torch.cat((torch.zeros(1, 6), torch.randn(4, 6) * 0.4)).double().requires_grad_()
        points = exp_map(tangent)
        self.assert_on_manifold(points)
        torch.testing.assert_close(log_map(points), tangent, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda x: log_map(exp_map(x)), (tangent,)))
        log_map(points).sum().backward()
        torch.testing.assert_close(tangent.grad, torch.ones_like(tangent))
        large = torch.tensor([[1e20, -1e20], [0., 0.]], requires_grad=True)
        bounded = log_map(exp_map(large))
        self.assert_on_manifold(exp_map(large))
        self.assertLessEqual(bounded.norm(dim=-1).max().item(), MAX_TANGENT_NORM + 1e-10)
        bounded.sum().backward()
        self.assertTrue(torch.isfinite(large.grad).all())

    def test_centroid_known_geodesic_and_gradient(self):
        # 同一测地线上的两个等权点，重心位于测地中点。
        points = exp_map(torch.tensor([[0.2, 0.], [1.4, 0.]], dtype=torch.float64))
        center = lorentz_centroid(points.mean(dim=0, keepdim=True))
        self.assert_on_manifold(center)
        torch.testing.assert_close(log_map(center), torch.tensor([[0.8, 0.]], dtype=torch.float64))
        opposite = exp_map(torch.tensor([[0.7, 0.], [-0.7, 0.]], dtype=torch.float64))
        torch.testing.assert_close(log_map(lorentz_centroid(opposite.mean(dim=0))), torch.zeros(2, dtype=torch.float64))
        tangent = (torch.randn(3, 4, dtype=torch.float64) * 0.3).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: log_map(lorentz_centroid(exp_map(x).mean(dim=0))), (tangent,)))
        # 非共线邻居的双曲重心不能退化为欧氏均值的映射包装。
        tangent = torch.tensor([[2., 0.], [0., 0.3]], dtype=torch.float64)
        center = log_map(lorentz_centroid(exp_map(tangent).mean(dim=0)))
        self.assertFalse(torch.allclose(center, tangent.mean(dim=0), atol=1e-3))

    def test_directed_edges_and_isolated_nodes(self):
        tangent = torch.tensor([[0.2, 0.], [1.4, 0.], [0., 1.], [0.3, 0.4]], dtype=torch.float64)
        # 0 -> 2 和 1 -> 2；不反向传播消息，不隐式加入自环。
        edges = torch.tensor([[0, 1], [2, 2]])
        conv = LorentzGNNConv()
        output = conv(exp_map(tangent), edges)
        self.assert_on_manifold(output)
        expected = torch.zeros_like(tangent)
        expected[2, 0] = 0.8
        torch.testing.assert_close(log_map(output), expected)
        empty = conv(exp_map(tangent), torch.empty((2, 0), dtype=torch.long))
        self.assert_on_manifold(empty)
        torch.testing.assert_close(log_map(empty), torch.zeros_like(tangent))

    def test_encoder_dropout_and_autocast(self):
        fixture = test_grid.GridTest()
        fixture.setUp()
        args = copy.deepcopy(fixture.args)
        args.dropout = 0.5
        encoder = LorentzGNNEncoder(args).train()
        feature = torch.randn(args.user_num, args.dim, requires_grad=True)
        layer_inputs = []
        handles = [layer.register_forward_pre_hook(lambda mod, inputs: layer_inputs.append(inputs[0]))
                   for layer in encoder.layers]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = encoder(feature, fixture.graph)
        for handle in handles:
            handle.remove()
        self.assertEqual(output.shape, feature.shape)
        self.assertEqual(output.dtype, feature.dtype)
        for points in layer_inputs:
            self.assert_on_manifold(points)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(encoder.layer_weights.grad).all())
        encoder.eval()
        torch.testing.assert_close(encoder(feature, fixture.graph), encoder(feature, fixture.graph), rtol=0, atol=0)

    def test_dataset_shapes_and_checkpoint(self):
        for dataset in ('twitter', 'douban', 'quora', 'weibo'):
            args = parse_args(['--dataset', dataset, '--gnn_type', 'lorentz_lightgcn'])
            setup_info(args)
            encoder = LorentzGNNEncoder(args)
            graph = Data(edge_index=torch.tensor([[1, 2, 3], [2, 3, 1]]))
            output = encoder(torch.randn(5, args.dim) * 0.1, graph)
            self.assertEqual(output.shape, (5, args.dim))
        fixture = test_grid.GridTest()
        fixture.setUp()
        args = fixture.args
        args.gnn_type = 'lorentz_lightgcn'
        model = GRID(args)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lorentz.pt'
            save_checkpoint(path, snapshot_state(model), args, 0, {})
            restored = GRID(args)
            load_checkpoint(path, restored, args)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            other = copy.deepcopy(args)
            other.gnn_type = 'lightgcn'
            with self.assertRaises(ValueError):
                load_checkpoint(path, GRID(other), other)


if __name__ == '__main__':
    unittest.main()
