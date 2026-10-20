"""可选的 Decoder 实验辅助逻辑：检查点与按轮汇总的诊断统计。"""

import logging
from pathlib import Path

import torch


MODEL_CONFIG_KEYS = ('dataset', 'dim', 'user_num', 'gnn_type', 'gnn_layers',
                     'clustering', 'group_num', 'n_heads', 'max_len', 'dropout',
                     'deterministic_eval', 'decoder_type', 'decoder_dim',
                     'decoder_weight', 'decoder_temperature')


def snapshot_state(model):
    # 必须复制张量；dict.copy() 会继续引用训练中的参数。
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
    has_decoder = any(name.startswith('Decoder.') for name in state)
    for key in MODEL_CONFIG_KEYS:
        if key not in config or not hasattr(args, key):
            continue
        if key in ('dropout', 'deterministic_eval', 'decoder_weight', 'decoder_temperature'):
            continue
        if key.startswith('decoder_') and not has_decoder:
            continue
        if config[key] != getattr(args, key):
            raise ValueError(f'Checkpoint configuration mismatch: {key}={config[key]}, requested {getattr(args, key)}')
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    allowed_missing = {k for k in expected if k.startswith('Decoder.')} if not has_decoder else set()
    if unexpected or missing != allowed_missing:
        raise ValueError(f'Checkpoint keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f'Checkpoint shape mismatch: {name}')
    model.load_state_dict(state, strict=not missing)
    logging.info('Loaded checkpoint %s; newly initialized keys: %s', path, sorted(missing))
    if not config:
        logging.warning('Legacy state_dict: tensor shapes checked; configuration metadata unavailable')
    if getattr(args, 'freeze_backbone', False) and not config.get('deterministic_eval', False):
        logging.warning('Frozen backbone uses deterministic grouping; compare its initial OFF scores, not historical stochastic evaluation')


class DecoderDiagnostics:
    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, name, value, count=1):
        if count:
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * count
            self.counts[name] = self.counts.get(name, 0) + count

    @torch.no_grad()
    def update(self, args, details, previous_mask, labels, label_mask, negatives, neg_mask, components, objective):
        batch_size = labels.size(0)
        # loss 是批均值，按样本数汇总；距离按有效节点事件数汇总。
        for name, value in components.items():
            self.add(name, value.detach().item(), batch_size)
        self.add('objective', objective.detach().item(), batch_size)
        eligible = previous_mask == 0
        eligible[:, 0] = False
        base = details['base_scores'].detach()
        decoder = details['decoder_scores']
        branches = {'base': base}
        if decoder is not None:
            decoder = decoder.detach()
            branches['weighted_decoder'] = args.decoder_weight * decoder
            distances = (-decoder * args.decoder_temperature).clamp_min(0).sqrt()
            for name, indices, valid in (('positive_distance', labels, label_mask),
                                         ('negative_distance', negatives, neg_mask)):
                valid = valid & (indices != 0) & eligible.gather(1, indices)
                values = distances.gather(1, indices)[valid]
                if values.numel():
                    self.add(name, values.mean().item(), values.numel())
        for name, scores in branches.items():
            # 在每条级联可预测候选集合内统计，不能把 -1e9 屏蔽值计入尺度。
            for row, valid in zip(scores, eligible):
                values = row[valid]
                if values.numel():
                    self.add(name + '_candidate_std', values.std(unbiased=False).item())
                    self.add(name + '_candidate_mean', values.mean().item())

    def summary(self):
        result = {k: self.totals[k] / self.counts[k] for k in self.totals}
        if 'negative_distance' in result and 'positive_distance' in result:
            result['distance_gap'] = result['negative_distance'] - result['positive_distance']
        return result
