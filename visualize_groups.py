"""从原版 GRID 检查点绘制一条测试级联的软分组。请在项目根目录运行。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import parse_args, setup_info
from dataLoader import load_social_graph
from model import GRID
from utils.Checkpoint import load_checkpoint


@torch.inference_mode()
def assignments(dataset, checkpoint, index):
    args = parse_args(['--dataset', dataset, '--deterministic_eval'])
    setup_info(args)
    args.device = 'cpu'
    model = GRID(args).eval()
    load_checkpoint(checkpoint, model, args)
    with open(args.cascade_path_test, encoding='utf-8') as f:
        cascades = json.load(f)
    if not 0 <= index < len(cascades):
        raise ValueError(f'cascade_index must be in [0, {len(cascades)-1}]')
    # 保留默认测试批次、截断和 PAD；原版 KMeans 的停止条件取整个批次的最大变化。
    start = index // args.batch_size * args.batch_size
    ids = torch.zeros(min(args.batch_size, len(cascades) - start), args.max_len, dtype=torch.long)
    for row, cascade in enumerate(cascades[start:start + args.batch_size]):
        prefix = cascade['observed'][:args.max_len]
        ids[row, :len(prefix)] = torch.tensor(prefix)
    embeddings = model.GNN(model.Embed.weight, load_social_graph(args))
    _, _, _, weights = model.Grouping(embeddings[ids], ids == 0, return_assignments=True)
    row = index - start
    valid = ids[row] != 0
    probs = weights[row, valid]
    if not len(probs) or not torch.allclose(probs.sum(1), torch.ones_like(probs[:, 0])):
        raise ValueError('Original grouping produced no valid assignments for this cascade/batch')
    return ids[row, valid].numpy(), probs.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='twitter')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cascade_index', type=int, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('analysis/cascade_groups'))
    opt = parser.parse_args()
    users, probs = assignments(opt.dataset, opt.checkpoint, opt.cascade_index)
    n, k = probs.shape
    dest = opt.output_dir / opt.dataset / opt.checkpoint.stem / f'test_{opt.cascade_index}'
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'assignments.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'user_id', *(f'G{i+1}' for i in range(k))])
        for pos, (user, row) in enumerate(zip(users, probs), 1):
            writer.writerow([pos, user, *row])
    fig, (ax, mass) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1], layout='constrained')
    heat = ax.imshow(probs, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0,
                     vmax=max(1/k, float(probs.max())))
    fig.colorbar(heat, ax=[ax, mass], label='Assignment probability')
    ticks = np.linspace(0, n-1, min(n, 14), dtype=int)
    ax.set_yticks(ticks, [f'{p+1}  |  {users[p]}' for p in ticks]); ax.set_ylabel('Position  |  user ID')
    ax.set_title(f'{opt.dataset} / test cascade {opt.cascade_index} / {opt.checkpoint.name}\n'
                 f'{n} observed positions   |   mean top probability {probs.max(1).mean():.3f}   |   uniform {1/k:.3f}')
    mass.bar(np.arange(k), probs.sum(0), color='#28666e')
    mass.axhline(n/k, color='#aa4931', ls='--', label='Uniform share'); mass.legend(loc='upper right')
    mass.set(xlim=(-0.5, k-0.5), xlabel='Group within this cascade', ylabel='Soft member count')
    mass.set_xticks(np.arange(k), [f'G{i+1}' for i in range(k)])
    # 共用同一颜色条列，保证上下两图的群编号对齐。
    ax.set_xticks(np.arange(k), [f'G{i+1}' for i in range(k)])
    fig.savefig(dest / 'groups.png', dpi=160); plt.close(fig)
    print(dest, 'mean max probability:', probs.max(1).mean())


if __name__ == '__main__':
    main()
