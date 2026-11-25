import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from module import GroupAttn, KMeansLayer, GeometricKMeansLayer, WeightedPooling, GraphEncoder, SelfAttn, PositionEmbedding

class GRID(nn.Module):
    def __init__(self, args):
        super(GRID, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num

        self.Embed = nn.Embedding(self.user_num, self.dim, padding_idx=0)
        self.GNN = GraphEncoder(args)

        if args.clustering:
            self.geometric_grouping = getattr(args, 'grouping_space', 'euclidean') != 'euclidean'
            self.Grouping = GeometricKMeansLayer(args) if self.geometric_grouping else KMeansLayer(args)
            self.GroupAttn = GroupAttn(args)
        else:
            self.Attn = SelfAttn(args.dim)

        self.Pooling = WeightedPooling(args)
        self.Predictor = nn.Linear(self.dim, self.user_num)
        self.reset_parameters()
        self.group_loss_weight = getattr(args, 'group_loss_weight', 0.0)
        if self.group_loss_weight > 0:
            if not args.clustering or not self.geometric_grouping:
                raise ValueError('Next-group prediction requires geometric grouping')
            self.GroupQuery = nn.Linear(self.dim, self.dim, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, args, cascade, cas_mask, previous_mask, graph, first_label=None, return_groups=False):
        predict_groups = return_groups or first_label is not None
        if predict_groups and self.group_loss_weight <= 0:
            raise ValueError('Set group_loss_weight > 0 to enable next-group prediction')
        initial_user = self.Embed.weight
        user_Embeddings = self.GNN(initial_user, graph)
        casEmbed = F.embedding(cascade.long(), user_Embeddings)

        if args.clustering:
            if predict_groups:
                casEmbed, cluster_mask, cluster_counts, centers = self.Grouping(casEmbed, cas_mask, return_centers=True)
            else:
                casEmbed, cluster_mask, cluster_counts = self.Grouping(casEmbed, cas_mask)
            # 新分组返回有效群掩码；注意力需要 PAD 掩码，池化需要有效群掩码。
            pad_mask = ~cluster_mask if self.geometric_grouping else cluster_mask
            h_c = self.GroupAttn(casEmbed, pad_mask, cluster_counts)
            h = self.Pooling(h_c, cluster_mask, cluster_counts)
        else:
            h_c = self.Attn(casEmbed, casEmbed, casEmbed, cas_mask)
            seq_lens = (cascade != 0).sum(dim=1).clamp(min=1) - 1
            h = h_c[torch.arange(h_c.size(0), device=h_c.device), seq_lens, :]

        pred_users = self.Predictor(h) + previous_mask

        if predict_groups:
            # 最后一位是占位群。预测只依赖当前前缀的级联和群体表示。
            logits = (self.GroupQuery(h).unsqueeze(1) * h_c[:, :-1]).sum(dim=-1) / sqrt(self.dim)
            active = cluster_mask[:, :-1]
            logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
            if first_label is None:
                return pred_users, logits
            with torch.no_grad():
                next_users = F.embedding(first_label.long(), user_Embeddings)
                target = self.Grouping.next_group_targets(next_users, centers, active, first_label)
            return pred_users, logits, target, cluster_counts[:, :-1].argmax(dim=1)
        return pred_users

