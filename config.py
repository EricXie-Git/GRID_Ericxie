import argparse
import math
from typing import List
from dataclasses import dataclass


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official implementation of GRID.")

    parser.add_argument("--seed", type=int, default=21, help="Random seed")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout probability used across model layers")

    data_group = parser.add_argument_group('Data Configs')
    data_group.add_argument('--dataset', type=str, default='twitter', choices=list(DATASET_CONFIGS.keys()), help="Dataset identifier used to resolve paths and dataset-specific defaults")
    data_group.add_argument('--batch_size', type=int, default=32, help="Mini-batch size for the DataLoaders")
    data_group.add_argument('--max_len', type=int, default=200, help="Maximum observed cascade length after truncation or padding")

    learning_group = parser.add_argument_group('Learning Configs')
    learning_group.add_argument('--max_epochs', type=int, default=30, help="Number of training epochs")
    learning_group.add_argument('--print_steps', type=int, default=10, help="Training steps between logging updates")
    learning_group.add_argument('--learning_rate', type=float, default=1e-3, help="Reserved learning-rate argument; the current optimizer setup uses a fixed value in utils/Optim.py")
    learning_group.add_argument('--weight_decay', type=float, default=1e-4, help="Reserved weight-decay argument; the current optimizer setup uses a fixed value in utils/Optim.py")

    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--clustering', type=bool, default=True, help="Apply grouping with K-Means before sequence attention")
    model_group.add_argument('--bpr_margin', type=float, default=0.5, help="Margin used by the BPR term in the hybrid loss")
    model_group.add_argument('--neg_num', type=int, default=100, help="Maximum number of negative samples drawn per cascade")
    model_group.add_argument('--gnn_type', type=str, default='lightgcn', choices=['gat', 'gcn', 'sage', 'lightgcn'], help="Backbone used by the social graph encoder")
    model_group.add_argument('--n_heads', type=int, default=6, help="Number of attention heads in the grouped transformer")
    model_group.add_argument('--group_num', type=int, default=20, help="Number of grouped tokens produced by clustering, including the padding group")

    decoder_group = parser.add_argument_group('Auxiliary Decoder')
    decoder_group.add_argument('--decoder_type', choices=['none', 'euclidean', 'hyperbolic'], default='none', help="Optional distance scoring branch")
    decoder_group.add_argument('--decoder_dim', type=int, default=32, help="Spatial dimension of the auxiliary decoder")
    decoder_group.add_argument('--decoder_weight', type=float, default=0.1, help="Nonnegative weight added to the original GRID scores")
    decoder_group.add_argument('--decoder_temperature', type=float, default=1.0, help="Positive temperature for negative squared distance scores")
    decoder_group.add_argument('--decoder_diagnostics', action='store_true', help="Log loss components, sampled distances and candidate score scales")
    decoder_group.add_argument('--compare_decoder', action='store_true', help="Evaluate fused and original scores in the same forward pass")
    decoder_group.add_argument('--freeze_backbone', action='store_true', help="Load a checkpoint and train only Decoder in a deterministic frozen backbone")
    decoder_group.add_argument('--decoder_objective', choices=['fused', 'bpr', 'separate'], default='fused', help="Fused loss, frozen decoder BPR, or separate GRID loss and decoder BPR")
    decoder_group.add_argument('--decoder_loss_weight', type=float, default=0.1, help="Decoder BPR weight in separate mode; independent of score fusion")
    decoder_group.add_argument('--decoder_detach_backbone', action='store_true', help="Block decoder gradients to the backbone in separate mode")
    decoder_group.add_argument('--deterministic_eval', action='store_true', help="Disable KMeans initialization noise during evaluation")
    learning_group.add_argument('--patience', type=int, default=5, help="Early stopping patience in epochs")
    learning_group.add_argument('--early_stop_after_warmup', action='store_true', help="Start counting early stopping patience after warmup")
    learning_group.add_argument('--selection_split', choices=['test', 'val'], default='test', help="Selection split; use val for new controlled experiments (test preserves legacy behaviour)")

    checkpoint_group = parser.add_argument_group('Checkpoint Configs')
    checkpoint_group.add_argument('--saved_model_path', type=str, default='checkpoint/', help="Directory created before training for checkpoint-related outputs")
    checkpoint_group.add_argument('--ckpt_file', type=str, default='checkpoint/model_.bin', help="Output path used with --save_checkpoint")
    checkpoint_group.add_argument('--save_checkpoint', action='store_true', help="Save best model weights and configuration; not optimizer resume state")
    checkpoint_group.add_argument('--load_checkpoint', type=str, default=None, help="Initialize from a trusted GRID checkpoint")
    checkpoint_group.add_argument('--best_score', type=float, default=0.0, help="Initial best-score threshold used for model selection")
    checkpoint_group.add_argument('--metric_k', type=List[int], default=[50, 100], help="Cutoff values used to report Recall@K and NDCG@K")
    checkpoint_group.add_argument('--epsilon', type=float, default=0.3, help="Reserved argument; not used by the current training or evaluation pipeline")

    args = parser.parse_args(argv)
    if args.decoder_dim < 1:
        parser.error('decoder_dim must be positive')
    if not math.isfinite(args.decoder_weight) or args.decoder_weight < 0:
        parser.error('decoder_weight must be finite and nonnegative')
    if not math.isfinite(args.decoder_temperature) or args.decoder_temperature <= 0:
        parser.error('decoder_temperature must be finite and positive')
    if args.patience < 1:
        parser.error('patience must be positive')
    if not math.isfinite(args.decoder_loss_weight) or args.decoder_loss_weight < 0:
        parser.error('decoder_loss_weight must be finite and nonnegative')
    if args.decoder_objective == 'separate' and (args.decoder_type == 'none' or args.freeze_backbone):
        parser.error('separate requires an enabled decoder and an unfrozen backbone')
    if args.decoder_detach_backbone and args.decoder_objective != 'separate':
        parser.error('decoder_detach_backbone requires separate mode')
    if args.freeze_backbone and (not args.load_checkpoint or args.decoder_type == 'none'):
        parser.error('freeze_backbone requires load_checkpoint and an enabled decoder')
    if args.decoder_objective == 'bpr' and not args.freeze_backbone:
        parser.error('decoder-only BPR requires freeze_backbone')
    if args.freeze_backbone and args.decoder_objective == 'fused' and args.decoder_weight == 0:
        parser.error('frozen fused training requires decoder_weight > 0')
    if args.freeze_backbone:
        args.deterministic_eval = True
    return args


def setup_info(args: argparse.Namespace) -> None:
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}."
                       f"Available datasets: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[args.dataset]
    args.user_num = config.user_num
    args.dim = config.dim
    args.n_warmup_steps = config.n_warmup_steps
    args.gnn_layers = config.gnn_layers

    base_path = f'dataset/{args.dataset}'
    args.cascade_path_train = f'{base_path}/cascade_train_neg.json'
    args.cascade_path_valid = f'{base_path}/cascade_val.json'
    args.cascade_path_test = f'{base_path}/cascade_test.json'
    args.graph_path = f'{base_path}/graph.pt'


@dataclass
class DatasetConfig:
    user_num: int
    dim: int
    n_warmup_steps: int
    gnn_layers: int


DATASET_CONFIGS = {
    'twitter': DatasetConfig(
        user_num=12627 + 1,
        dim=128,
        n_warmup_steps=500,
        gnn_layers=2
    ),
    'douban': DatasetConfig(
        user_num=12232 + 1,
        dim=128,
        n_warmup_steps=500,
        gnn_layers=2
    ),
    'quora': DatasetConfig(
        user_num=4578 + 1,
        dim=128,
        n_warmup_steps=100,
        gnn_layers=1
    ),
    'weibo': DatasetConfig(
        user_num=31061 + 1,
        dim=256,
        n_warmup_steps=500,
        gnn_layers=2
    )
}
