import argparse
from typing import List
from dataclasses import dataclass


def parse_args() -> argparse.Namespace:
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

    checkpoint_group = parser.add_argument_group('Checkpoint Configs')
    checkpoint_group.add_argument('--saved_model_path', type=str, default='checkpoint/', help="Directory created before training for checkpoint-related outputs")
    checkpoint_group.add_argument('--ckpt_file', type=str, default='checkpoint/model_.bin', help="Reserved checkpoint file path; the current training loop keeps the best model state in memory")
    checkpoint_group.add_argument('--best_score', type=float, default=0.0, help="Initial best-score threshold used for model selection")
    checkpoint_group.add_argument('--metric_k', type=List[int], default=[50, 100], help="Cutoff values used to report Recall@K and NDCG@K")
    checkpoint_group.add_argument('--epsilon', type=float, default=0.3, help="Reserved argument; not used by the current training or evaluation pipeline")

    return parser.parse_args()


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
