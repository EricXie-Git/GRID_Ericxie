"""GRID 小图回归检查，不运行真实数据集实验。"""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

import main
from config import parse_args
from dataLoader import batch_process
from model import GRID
from module import Combined_Loss
from utils.Checkpoint import snapshot_state, save_checkpoint, load_checkpoint
from utils.Optim import build_optimizer


class GridTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.args = SimpleNamespace(dim=16, user_num=15, dropout=0.0, clustering=True,
                                    group_num=4, n_heads=4, max_len=8, gnn_type='lightgcn',
                                    gnn_layers=2, bpr_margin=0.5, dataset='toy', device='cpu',
                                    deterministic_eval=True, metric_k=[3, 5], n_warmup_steps=0,
                                    max_epochs=2, best_score=0.0, print_steps=10,
                                    selection_split='val', patience=2, save_checkpoint=True)
        self.batch = {'observed': torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
                      'labels': torch.tensor([[8, 9], [9, 10]]), 'label_count': torch.tensor([2, 2]),
                      'negative_samples': torch.tensor([[11, 12], [12, 13]]), 'neg_count': torch.tensor([2, 2])}
        source = torch.arange(1, 15)
        target = source.roll(1)
        self.graph = Data(edge_index=torch.stack((torch.cat((source, target)),
                                                 torch.cat((target, source)))), num_nodes=15)

    def test_original_config_and_removed_flags(self):
        args = parse_args([])
        self.assertEqual(args.gnn_type, 'lightgcn')
        self.assertEqual(args.bpr_margin, 0.5)
        for flags in (['--decoder_type', 'hyperbolic'], ['--freeze_backbone'],
                      ['--decoder_objective', 'separate'], ['--compare_decoder']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_forward_backward_and_graph_encoders(self):
        for gnn_type in ('lightgcn', 'gat', 'gcn', 'sage', 'lorentz_lightgcn'):
            for clustering in (True, False):
                with self.subTest(gnn_type=gnn_type, clustering=clustering):
                    args = copy.deepcopy(self.args)
                    args.gnn_type, args.clustering = gnn_type, clustering
                    model = GRID(args)
                    cascade, mask, labels, lm, negatives, nm, first, previous = batch_process(args, self.batch)
                    scores = model(args, cascade, mask, previous, self.graph)
                    self.assertEqual(scores.shape, (2, 15))
                    self.assertTrue((scores[previous < 0] < -1e8).all())
                    loss = Combined_Loss(args, scores, labels, lm, negatives, nm, first)
                    self.assertTrue(torch.isfinite(loss))
                    before = model.Predictor.weight.detach().clone()
                    optimizer, scheduler = build_optimizer(args, model, 2)
                    loss.backward()
                    self.assertGreater(model.Embed.weight.grad.abs().sum().item(), 0)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                    optimizer.step()
                    scheduler.step()
                    self.assertFalse(torch.equal(before, model.Predictor.weight))

    def test_checkpoint_roundtrip_and_validation(self):
        args = self.args
        model = GRID(args)
        state = snapshot_state(model)
        with torch.no_grad():
            model.Predictor.weight.add_(1)
        self.assertFalse(torch.equal(state['Predictor.weight'], model.Predictor.weight))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pt'
            save_checkpoint(path, state, args, 0, {})
            load_checkpoint(path, model, args)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            # 兼容原始 state_dict；缺键和多余参数不应被静默忽略。
            torch.save(state, path)
            load_checkpoint(path, model, args)
            broken = dict(state)
            broken.pop('Embed.weight')
            torch.save(broken, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, args)
            broken = dict(state, unexpected=torch.zeros(1))
            torch.save(broken, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, args)

    def test_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            for gnn_type, selection_split in (('lightgcn', 'test'), ('lightgcn', 'val'),
                                               ('lorentz_lightgcn', 'test'), ('lorentz_lightgcn', 'val')):
                args = copy.deepcopy(self.args)
                args.gnn_type = gnn_type
                args.selection_split = selection_split
                args.ckpt_file = str(Path(temp) / 'model.pt')
                with patch.object(main, 'create_dataloaders', return_value=([self.batch], [self.batch], [self.batch])), \
                     patch.object(main, 'load_social_graph', return_value=self.graph), \
                     contextlib.redirect_stdout(io.StringIO()):
                    main.train(args)
                    args.load_checkpoint = args.ckpt_file
                    main.train(args)
                checkpoint = torch.load(args.ckpt_file, weights_only=True)
                self.assertEqual(checkpoint['selection_split'], selection_split)
                self.assertEqual(set(main.best_scores), {'recall@3', 'recall@5', 'ndcg@3', 'ndcg@5'})


if __name__ == '__main__':
    unittest.main()
