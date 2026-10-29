"""通用模型检查点：保存最佳权重并校验加载配置，不恢复优化器状态。"""

import logging
from pathlib import Path

import torch


MODEL_CONFIG_KEYS = ('dataset', 'dim', 'user_num', 'gnn_type', 'gnn_layers',
                     'clustering', 'group_num', 'n_heads', 'max_len', 'dropout',
                     'deterministic_eval')


def snapshot_state(model):
    # 复制实际张量，避免最佳权重随后续训练一起变化。
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(path, state, args, epoch, scores):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': state,
                'model_config': {k: getattr(args, k) for k in MODEL_CONFIG_KEYS if hasattr(args, k)},
                'epoch': epoch, 'selection_split': getattr(args, 'selection_split', 'test'),
                'selection_scores': scores}, path)


def load_checkpoint(path, model, args):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state', checkpoint)
    config = checkpoint.get('model_config', {})
    for key in MODEL_CONFIG_KEYS:
        if key not in config or not hasattr(args, key) or key in ('dropout', 'deterministic_eval'):
            continue
        if config[key] != getattr(args, key):
            raise ValueError(f'Checkpoint configuration mismatch: {key}={config[key]}, requested {getattr(args, key)}')
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or missing:
        raise ValueError(f'Checkpoint keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f'Checkpoint shape mismatch: {name}')
    model.load_state_dict(state)
    logging.info('Loaded checkpoint %s', path)
    if not config:
        logging.warning('Legacy state_dict: tensor shapes checked; configuration metadata unavailable')
