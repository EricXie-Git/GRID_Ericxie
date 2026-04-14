import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from module import GroupAttn, KMeansLayer, WeightedPooling, GraphEncoder, SelfAttn, PositionEmbedding

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
            h = h_c[torch.arange(h_c.size(0)), seq_lens, :]

        pred_users = self.Predictor(h) + previous_mask

        return pred_users

