import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import math
from contextlib import nullcontext

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
        self.backbone_frozen = False

    def freeze_backbone(self):
        if self.Decoder is None:
            raise ValueError('Freezing requires an enabled Decoder')
        self.backbone_frozen = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('Decoder.'))
        if hasattr(self, 'Grouping'):
            self.Grouping.deterministic_eval = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            # 外部调用 model.train() 时仍保持主干 dropout/归一化层为评估模式。
            for name, child in self.named_children():
                if name != 'Decoder':
                    child.eval()
        return self

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def encode(self, args, cascade, cas_mask, graph):
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

        return h, user_Embeddings, self.Predictor(h)

    def forward(self, args, cascade, cas_mask, previous_mask, graph, return_details=False):
        with torch.no_grad() if self.backbone_frozen else nullcontext():
            h, user_Embeddings, base_scores = self.encode(args, cascade, cas_mask, graph)
        decoder_scores = None
        if self.Decoder is not None and (self.decoder_weight > 0 or return_details):
            # 仅隔离分支梯度；主干仍由原 GRID 损失正常训练。
            if getattr(args, 'decoder_detach_backbone', False):
                decoder_scores = self.Decoder(h.detach(), user_Embeddings.detach())
            else:
                decoder_scores = self.Decoder(h, user_Embeddings)
        pred_users = base_scores
        if decoder_scores is not None and self.decoder_weight > 0:
            pred_users = pred_users + self.decoder_weight * decoder_scores
        # 融合之后统一应用已有的 PAD/已参与用户屏蔽，损失接口保持不变。
        pred_users = pred_users + previous_mask

        if return_details:
            return pred_users, {'base_scores': base_scores, 'decoder_scores': decoder_scores}
        return pred_users

