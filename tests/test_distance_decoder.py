"""小图验证评分分支，不运行数据集训练。python -m unittest discover -s tests -v"""

import copy
import contextlib
import io
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Data

from config import parse_args
from model import GRID
from module import Combined_Loss, DistanceDecoder


def make_args(decoder_type='none', clustering=True, gnn_type='lightgcn'):
    return SimpleNamespace(dim=16, user_num=15, dropout=0.0, clustering=clustering,
                           group_num=4, n_heads=4, max_len=8, gnn_type=gnn_type,
                           gnn_layers=2, bpr_margin=0.5, decoder_type=decoder_type,
                           decoder_dim=6, decoder_weight=0.1, decoder_temperature=1.0)


class DistanceDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.cascade = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        self.mask = self.cascade == 0
        self.previous_mask = torch.zeros(2, 15)
        self.previous_mask.scatter_(1, self.cascade, -1e9)
        self.previous_mask[:, 0] = -1e9
        source = torch.arange(1, 15)
        target = source.roll(1)
        self.graph = Data(edge_index=torch.stack((torch.cat((source, target)),
                                                 torch.cat((target, source)))), num_nodes=15)

    def test_config(self):
        self.assertEqual(parse_args([]).decoder_type, 'none')
        for flags in (['--decoder_dim', '0'], ['--decoder_weight', '-1'],
                      ['--decoder_temperature', '0'], ['--decoder_temperature', 'nan']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(flags)

    def test_old_args_and_checkpoint(self):
        args = make_args()
        old_args = copy.deepcopy(args)
        for name in ('decoder_type', 'decoder_dim', 'decoder_weight', 'decoder_temperature'):
            delattr(old_args, name)
        torch.manual_seed(10)
        original = GRID(old_args).eval()
        torch.manual_seed(10)
        disabled = GRID(args).eval()
        disabled.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(any(name.startswith('Decoder.') for name in disabled.state_dict()))
        torch.manual_seed(11)
        expected = original(old_args, self.cascade, self.mask, self.previous_mask, self.graph)
        torch.manual_seed(11)
        actual = disabled(args, self.cascade, self.mask, self.previous_mask, self.graph)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        args.decoder_type = 'hyperbolic'
        args.decoder_weight = 0.0
        torch.manual_seed(10)
        enabled = GRID(args).eval()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(enabled.state_dict()[name], value, rtol=0, atol=0)
        torch.manual_seed(11)
        actual = enabled(args, self.cascade, self.mask, self.previous_mask, self.graph)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_distance_and_lorentz_constraint(self):
        decoder = DistanceDecoder(make_args('hyperbolic'))
        tangent = torch.zeros(2, 6, requires_grad=True)
        tangent.data[1, 0] = 1.0
        points = decoder.exp_map(tangent)
        torch.testing.assert_close(-points[:, 0].square() + points[:, 1:].square().sum(-1),
                                   -torch.ones(2), atol=1e-6, rtol=1e-6)
        distance = decoder.squared_distance(points, points)
        torch.testing.assert_close(distance, torch.tensor([[0., 1.], [1., 0.]]), atol=2e-6, rtol=2e-6)
        distance.sum().backward()
        self.assertTrue(torch.isfinite(tangent.grad).all())

    def test_chunking_zero_large_inputs_and_autocast(self):
        for mode in ('euclidean', 'hyperbolic'):
            decoder = DistanceDecoder(make_args(mode))
            query = torch.randn(2, 16, requires_grad=True)
            users = torch.randn(15, 16, requires_grad=True)
            expected = decoder(query, users)
            decoder.chunk_size = 4
            torch.testing.assert_close(decoder(query, users), expected, atol=1e-5, rtol=1e-5)
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                actual = decoder(query, users)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for scale in (0.0, 10000.0):
                q = (torch.randn(1, 16) * scale).requires_grad_()
                u = (torch.randn(1, 16) * scale).requires_grad_()
                score = decoder(q, u)
                score.sum().backward()
                self.assertTrue(torch.isfinite(score).all())
                self.assertTrue(torch.isfinite(q.grad).all())
                self.assertTrue(torch.isfinite(u.grad).all())

    def test_grid_loss_backward_and_optimizer(self):
        for clustering in (True, False):
            for mode in ('none', 'euclidean', 'hyperbolic'):
                with self.subTest(clustering=clustering, mode=mode):
                    args = make_args(mode, clustering)
                    model = GRID(args)
                    scores = model(args, self.cascade, self.mask, self.previous_mask, self.graph)
                    self.assertEqual(scores.shape, (2, 15))
                    self.assertTrue(torch.isfinite(scores).all())
                    self.assertTrue((scores[self.previous_mask < 0] < -1e8).all())
                    labels = torch.tensor([[8, 9], [9, 10]])
                    negatives = torch.tensor([[11, 12], [12, 13]])
                    loss = Combined_Loss(args, scores, labels, torch.ones_like(labels, dtype=torch.bool),
                                         negatives, torch.ones_like(negatives, dtype=torch.bool), labels[:, 0])
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if model.Decoder is not None:
                        for parameter in model.Decoder.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    torch.optim.Adam(model.parameters(), lr=1e-3).step()
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_graph_encoder_variants(self):
        for gnn in ('gcn', 'gat', 'sage'):
            args = make_args('hyperbolic', gnn_type=gnn)
            model = GRID(args)
            scores = model(args, self.cascade, self.mask, self.previous_mask, self.graph)
            scores[:, 8:].sum().backward()
            self.assertTrue(torch.isfinite(scores).all())
            self.assertTrue(torch.isfinite(model.Decoder.W_u.weight.grad).all())

    def test_dataset_dimensions(self):
        # 覆盖实际数据集的维度和用户规模，验证默认分块路径。
        for dim, users in ((128, 12628), (256, 31062)):
            args = make_args('hyperbolic')
            args.dim, args.decoder_dim = dim, 32
            decoder = DistanceDecoder(args)
            query = torch.randn(2, dim, requires_grad=True)
            embeddings = torch.randn(users, dim, requires_grad=True)
            scores = decoder(query, embeddings)
            self.assertEqual(scores.shape, (2, users))
            scores.mean().backward()
            self.assertTrue(torch.isfinite(embeddings.grad).all())
            self.assertTrue(torch.isfinite(query.grad).all())


if __name__ == '__main__':
    unittest.main()
