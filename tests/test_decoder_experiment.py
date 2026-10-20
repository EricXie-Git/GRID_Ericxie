"""验证冻结/检查点/诊断开关；仅使用小图和少量合成 batch。"""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

import main
from config import parse_args
from dataLoader import batch_process
from model import GRID
from module import Combined_Loss, BPR_Loss
from utils.DecoderExperiment import snapshot_state, save_checkpoint, load_checkpoint, DecoderDiagnostics
from utils.Optim import build_optimizer
from test_distance_decoder import make_args


class DecoderExperimentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.args = make_args()
        self.args.__dict__.update(dataset='toy', device='cpu', deterministic_eval=True,
                                  metric_k=[3, 5], n_warmup_steps=0, max_epochs=2,
                                  best_score=0.0, print_steps=10, selection_split='val',
                                  patience=2, save_checkpoint=True, decoder_diagnostics=True,
                                  compare_decoder=True, decoder_objective='fused')
        self.batch = {'observed': torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
                      'labels': torch.tensor([[8, 9], [9, 10]]), 'label_count': torch.tensor([2, 2]),
                      'negative_samples': torch.tensor([[11, 12], [12, 13]]), 'neg_count': torch.tensor([2, 2])}
        self.graph = Data(edge_index=torch.tensor([[1, 2, 2, 3, 4, 5, 5, 6],
                                                  [2, 1, 3, 2, 5, 4, 6, 5]]), num_nodes=15)

    def test_flags(self):
        flags = ['--freeze_backbone', '--load_checkpoint', 'base.pt', '--decoder_type', 'hyperbolic']
        self.assertTrue(parse_args(flags).deterministic_eval)
        for flags in (['--freeze_backbone'], ['--decoder_objective', 'bpr'], ['--patience', '0']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(flags)

    def test_frozen_updates_and_reproducibility(self):
        for objective in ('fused', 'bpr'):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as temp:
                baseline = GRID(self.args).eval()
                path = Path(temp) / 'base.pt'
                state = snapshot_state(baseline)
                save_checkpoint(path, state, self.args, 0, {})
                args = copy.deepcopy(self.args)
                args.decoder_type = 'hyperbolic'
                args.freeze_backbone = True
                model = GRID(args)
                load_checkpoint(path, model, args)
                model.freeze_backbone()
                model.train()
                self.assertFalse(model.GNN.training)
                self.assertFalse(model.Grouping.training)
                optimizer, _ = build_optimizer(args, model, 2)
                self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                                 {id(p) for p in model.Decoder.parameters()})
                cascade, mask, labels, lm, negatives, nm, first, previous = batch_process(args, self.batch)
                expected = baseline(self.args, cascade, mask, previous, self.graph)
                scores, details = model(args, cascade, mask, previous, self.graph, return_details=True)
                torch.testing.assert_close(details['base_scores'] + previous, expected, rtol=0, atol=0)
                loss, components = Combined_Loss(args, scores, labels, lm, negatives, nm, first, return_components=True)
                torch.testing.assert_close(loss, sum(components[k] for k in ('weighted_bce', 'weighted_bpr', 'weighted_ce')))
                diagnostics = DecoderDiagnostics()
                diagnostics.update(args, details, previous, labels, lm, negatives, nm, components, loss)
                values = diagnostics.summary()
                self.assertIn('distance_gap', values)
                self.assertLess(values['base_candidate_std'], 100)
                if objective == 'bpr':
                    loss = BPR_Loss(args, details['decoder_scores'], labels, lm, negatives, nm)
                before_decoder = model.Decoder.W_u.weight.detach().clone()
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before_decoder, model.Decoder.W_u.weight))
                for name, value in state.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                _, after = model(args, cascade, mask, previous, self.graph, return_details=True)
                torch.testing.assert_close(after['base_scores'], details['base_scores'], rtol=0, atol=0)
                on, off = main.inference(args, model, [self.batch], self.graph, compare_decoder=True)
                self.assertEqual(off, main.inference(self.args, baseline, [self.batch], self.graph))
                # 完整分支检查点可严格重载；主干缺键必须报错。
                full = Path(temp) / 'full.pt'
                save_checkpoint(full, snapshot_state(model), args, 1, on)
                reloaded = GRID(args)
                load_checkpoint(full, reloaded, args)
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, reloaded.state_dict()[k])
                broken = dict(state)
                broken.pop('Embed.weight')
                torch.save(broken, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path, reloaded, args)

    def test_main_training_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline_path = str(Path(temp) / 'baseline.pt')
            with patch.object(main, 'create_dataloaders', return_value=([self.batch], [self.batch], [self.batch])), \
                 patch.object(main, 'load_social_graph', return_value=self.graph), \
                 contextlib.redirect_stdout(io.StringIO()):
                args = copy.deepcopy(self.args)
                args.ckpt_file = baseline_path
                main.train(args)
                self.assertTrue(Path(baseline_path).is_file())
                checkpoint = torch.load(baseline_path, weights_only=True)
                self.assertEqual(checkpoint['selection_split'], 'val')
                for objective in ('fused', 'bpr'):
                    args.decoder_type = 'euclidean'
                    args.freeze_backbone = True
                    args.load_checkpoint = baseline_path
                    args.decoder_objective = objective
                    args.ckpt_file = str(Path(temp) / (objective + '.pt'))
                    main.train(args)
                    saved = torch.load(args.ckpt_file, weights_only=True)['model_state']
                    for name, value in checkpoint['model_state'].items():
                        torch.testing.assert_close(saved[name], value, rtol=0, atol=0)

    def test_diagnostic_distance_values(self):
        args = self.args
        args.decoder_weight = 0.1
        details = {'base_scores': torch.zeros(1, 4), 'decoder_scores': torch.tensor([[0., -1., -4., -9.]])}
        collector = DecoderDiagnostics()
        collector.update(args, details, torch.tensor([[-1e9, 0., 0., 0.]]),
                         torch.tensor([[1, 0]]), torch.tensor([[True, False]]),
                         torch.tensor([[2, 3]]), torch.tensor([[True, True]]), {}, torch.tensor(0.))
        values = collector.summary()
        self.assertEqual(values['positive_distance'], 1.0)
        self.assertEqual(values['negative_distance'], 2.5)
        self.assertEqual(values['distance_gap'], 1.5)


if __name__ == '__main__':
    unittest.main()
