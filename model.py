import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import math

from module import GroupAttn, KMeansLayer, WeightedPooling, GraphEncoder, SelfAttn, PositionEmbedding, DistanceDecoder

class GRID(nn.Module):
    def __init__(self, args):
        super(GRID, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num

        self.Embed = nn.Embedding(self.user_num, self.dim, padding_idx=0)
        self.GNN = GraphEncoder(args)

        if args.clustering:
            self.Grouping = KMeansLayer(args)
            self.GroupAttn = GroupAttn(args)
        else:
            self.Attn = SelfAttn(args.dim)

        self.Pooling = WeightedPooling(args)
        self.Predictor = nn.Linear(self.dim, self.user_num)
        self.reset_parameters()

        self.decoder_type = getattr(args, 'decoder_type', 'none')
        self.decoder_weight = getattr(args, 'decoder_weight', 0.1)
        if self.decoder_type not in ('none', 'euclidean', 'hyperbolic'):
            raise ValueError(f'Unsupported decoder type: {self.decoder_type}')
        if not math.isfinite(self.decoder_weight) or self.decoder_weight < 0:
            raise ValueError('decoder_weight must be finite and nonnegative')
        # 在主干初始化之后创建，保持相同随机种子下主干初值一致。
        # 默认不注册新参数，兼容原始检查点和旧的 args 对象。
        self.Decoder = DistanceDecoder(args) if self.decoder_type != 'none' else None

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, args, cascade, cas_mask, previous_mask, graph):
        initial_user = self.Embed.weight
        user_Embeddings = self.GNN(initial_user, graph)
        casEmbed = F.embedding(cascade.long(), user_Embeddings)

        if args.clustering:
            casEmbed, cluster_mask, cluster_counts = self.Grouping(casEmbed, cas_mask)
            h_c = self.GroupAttn(casEmbed, cluster_mask, cluster_counts)
            h = self.Pooling(h_c, cluster_mask, cluster_counts)
        else:
            h_c = self.Attn(casEmbed, casEmbed, casEmbed, cas_mask)
            seq_lens = (cascade != 0).sum(dim=1).clamp(min=1) - 1
            h = h_c[torch.arange(h_c.size(0), device=h_c.device), seq_lens, :]

        pred_users = self.Predictor(h)
        if self.Decoder is not None and self.decoder_weight > 0:
            pred_users = pred_users + self.decoder_weight * self.Decoder(h, user_Embeddings)
        # 融合之后统一应用已有的 PAD/已参与用户屏蔽，损失接口保持不变。
        pred_users = pred_users + previous_mask

        return pred_users

