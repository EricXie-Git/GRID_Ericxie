"""级联内的几何分组约束，共用分组层的距离与温度。"""
import math

import torch
from torch import nn


class GroupStructureLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_separation = getattr(args, 'group_sep_weight', 0.0) > 0
        self.use_preference = getattr(args, 'group_pref_weight', 0.0) > 0
        self.scale = getattr(args, 'group_sep_scale', 0.5)
        self.ratio = getattr(args, 'group_pref_ratio', 2.0)

    @staticmethod
    def preference_gap(distances, active, mask, temperature):
        # 使用平方距离差直接计算 log(p1/p2)，避免极小概率下溢。
        valid = ~mask & (active.sum(1, keepdim=True) >= 2) & ((~mask).sum(1, keepdim=True) >= 2)
        if distances.size(-1) < 2:
            return distances[..., 0] * 0, valid
        nearest = distances.masked_fill(~active.unsqueeze(1), torch.inf).topk(2, largest=False).values
        nearest = nearest.masked_fill(~valid.unsqueeze(-1), 0)
        return (nearest[..., 1] - nearest[..., 0]) / temperature, valid

    @torch.no_grad()
    def diagnostics(self, distances, active, mask, temperature):
        gap, valid = self.preference_gap(distances, active, mask, temperature)
        return torch.stack(((valid & (gap >= math.log(self.ratio))).sum(), valid.sum()))

    def forward(self, points, centers, active, mask, temperature, squared_distances):
        distances = squared_distances(points, centers)
        zero = distances.sum() * 0
        separation, preference = zero, zero
        if self.use_separation:
            with torch.no_grad():
                # 至少一个真实用户把该群选为首选，才认为这个群有成员支持。
                winner = distances.masked_fill(~active.unsqueeze(1), torch.inf).argmin(-1)
                hits = torch.zeros_like(active, dtype=torch.long).scatter_add_(1, winner, (~mask).long())
                supported = active & (hits > 0)
                pairs = (supported.unsqueeze(2) & supported.unsqueeze(1)).triu(1)
                user_pairs = (~mask.unsqueeze(2) & ~mask.unsqueeze(1)).triu(1)
                user_dist = squared_distances(points, points).clamp_min(0).sqrt()
                mean_dist = user_dist.masked_fill(~user_pairs, 0).sum((1, 2)) / user_pairs.sum((1, 2)).clamp_min(1)
                margin = self.scale * mean_dist[:, None, None]
            # 下界避免对重合点的距离开方时出现无穷梯度。
            spacing = squared_distances(centers, centers).clamp_min(1e-16).sqrt()
            penalties = (margin - spacing).clamp_min(0).square()
            separation = penalties.masked_fill(~pairs, 0).sum() / pairs.sum().clamp_min(1)
        if self.use_preference:
            gap, valid = self.preference_gap(distances, active, mask, temperature)
            penalties = (math.log(self.ratio) - gap).clamp_min(0)
            preference = penalties.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
        return separation, preference
