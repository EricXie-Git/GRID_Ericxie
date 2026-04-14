import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv


class GroupAttn(nn.Module):
    def __init__(self, args):
        super(GroupAttn, self).__init__()
        self.PE = PositionEmbedding(args)
        self.encoder = GroupTrm(input_size=args.dim, n_heads=args.n_heads, attn_dropout=args.dropout)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, seq_h, pad_mask, cluster_counts):
        seq_h = self.PE(seq_h) + seq_h
        seq_h = self.encoder(self.drop(seq_h), seq_h, seq_h, pad_mask, cluster_counts)
        return self.drop(seq_h)


class GroupTrm(nn.Module):
    def __init__(self, input_size, n_heads=4, d_k=None, d_v=None, is_layer_norm=True, attn_dropout=0.1):
        super(GroupTrm, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size // n_heads
        self.d_v = d_v if d_v is not None else input_size // n_heads
        self.input_size = input_size

        self.is_layer_norm = is_layer_norm

        self.W_q = nn.Linear(input_size, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(input_size, n_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(input_size, n_heads * self.d_v, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_v, input_size, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(input_size * 4, input_size)
        )

        if self.is_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size)
            self.layer_norm2 = nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q.weight)
        init.xavier_normal_(self.W_k.weight)
        init.xavier_normal_(self.W_v.weight)
        init.xavier_normal_(self.W_o.weight)

    def scaled_dot_product_attention(self, Q, K, V, attn_mask, cluster_counts=None, epsilon=1e-6):
        temperature = self.d_k ** 0.5
        scores = torch.bmm(Q, K.transpose(-2, -1)) / (temperature + epsilon)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        if cluster_counts is not None:
            max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
            scores_stable = scores - max_scores

            if torch.abs(scores_stable).max() > 20.0:
                scores_stable = torch.clamp(scores_stable, min=-20.0, max=20.0)

            exp_scores = torch.exp(scores_stable)
            weighted_exp_scores = cluster_counts * exp_scores
            attn_weights = weighted_exp_scores / (weighted_exp_scores.sum(dim=-1, keepdim=True) + epsilon)
        else:
            attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, V)
        return output

    def forward(self, Q, K, V, pad_mask, cluster_counts=None):
        bsz, q_len, _ = Q.size()
        k_len = K.size(1)

        residual = Q

        q_s = self.W_q(Q).view(bsz, q_len, self.n_heads, self.d_k).permute(0, 2, 1, 3).contiguous().view(-1, q_len, self.d_k)
        k_s = self.W_k(K).view(bsz, k_len, self.n_heads, self.d_k).permute(0, 2, 1, 3).contiguous().view(-1, k_len, self.d_k)
        v_s = self.W_v(V).view(bsz, k_len, self.n_heads, self.d_v).permute(0, 2, 1, 3).contiguous().view(-1, k_len, self.d_v)

        causal_mask = torch.triu(torch.ones(q_len, k_len, dtype=torch.bool, device=Q.device), diagonal=1)
        if pad_mask is not None:
            attn_mask = pad_mask.unsqueeze(1).expand(-1, q_len, -1) | causal_mask
        else:
            attn_mask = causal_mask

        attn_mask = attn_mask.unsqueeze(1).expand(bsz, self.n_heads, q_len, k_len).reshape(-1, q_len, k_len)

        if cluster_counts is not None:
            cluster_counts = cluster_counts.unsqueeze(1).expand(-1, self.n_heads, -1).reshape(bsz * self.n_heads,
                                                                                              -1).unsqueeze(1)

        V_att = self.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask, cluster_counts)

        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v).permute(0, 2, 1, 3).contiguous().view(bsz, q_len, -1)
        output = self.W_o(V_att)
        output = self.dropout(output)

        if self.is_layer_norm:
            output = self.layer_norm1(residual + output)
        else:
            output = residual + output
        residual2 = output

        output = self.ffn(output)
        output = self.dropout(output)

        if self.is_layer_norm:
            output = self.layer_norm2(residual2 + output)
        else:
            output = residual2 + output

        return output


class KMeansLayer(nn.Module):
    """
    Soft K-Means clustering layer for sequences.
    """

    def __init__(self, args):
        super(KMeansLayer, self).__init__()
        self.dim = args.dim
        self.cluster_num = args.group_num
        self.max_iter = 10
        self.temperature = 1.0
        self.eps = 1e-6

        self.max_clusters = self.cluster_num - 1
        self.cluster_centers = nn.Parameter(torch.randn(self.max_clusters, self.dim) * 0.1)
        nn.init.xavier_uniform_(self.cluster_centers)

        self.layer_norm = nn.LayerNorm(self.dim)

    def compute_distances(self, x, centers):
        """
        Compute squared Euclidean distances via ||x-c||^2 = ||x||^2 - 2<x,c> + ||c||^2.
        Avoids materialising the (bs, seq_len, K, dim) intermediate tensor.
        x: (bs, seq_len, dim)
        centers: (bs, max_clusters, dim)
        """
        x_sq = (x ** 2).sum(-1, keepdim=True)
        c_sq = (centers ** 2).sum(-1).unsqueeze(1)
        cross = torch.bmm(x, centers.transpose(1, 2))
        distances = x_sq + c_sq - 2 * cross
        distances.clamp_(min=0.0, max=100.0)
        return distances

    def forward(self, x, mask):
        """
        x: (bs, seq_len, dim)
        mask: (bs, seq_len) bool (True for PAD positions)
        returns:
            clustered_x: (bs, cluster_num, dim)
            cluster_mask: (bs, cluster_num) bool
            cluster_counts: (bs, cluster_num) float
        """
        x = self.layer_norm(x)
        bs, seq_len, dim = x.shape
        device = x.device

        valid_positions_mask = ~mask

        num_valid_tokens = valid_positions_mask.sum(dim=1)
        if num_valid_tokens.min() == 0:
            clustered_x = torch.zeros(bs, self.cluster_num, dim, device=device, dtype=x.dtype)
            cluster_mask = torch.zeros(bs, self.cluster_num, device=device, dtype=torch.bool)
            cluster_counts = torch.zeros(bs, self.cluster_num, device=device, dtype=x.dtype)
            return clustered_x, cluster_mask, cluster_counts

        with torch.no_grad():
            centers = self.cluster_centers.unsqueeze(0).expand(bs, -1, -1).clone()

            noise_scale = min(x.std().item() * 0.01, 0.01)
            centers.add_(noise_scale * torch.randn_like(centers))

            for iteration in range(self.max_iter):
                distances = self.compute_distances(x, centers)
                distances.masked_fill_(mask.unsqueeze(-1), 1e6)
                soft_assign = F.softmax(-distances / self.temperature, dim=-1)

                counts = soft_assign.sum(dim=1)
                weighted_sum = torch.bmm(soft_assign.transpose(1, 2), x)
                new_centers = weighted_sum / (counts.unsqueeze(-1) + self.eps)

                old_centers = centers
                if iteration > 0:
                    centers = 0.2 * centers + 0.8 * new_centers
                else:
                    centers = new_centers

                if iteration > 0:
                    center_diff = torch.norm(centers - old_centers, dim=-1).max()
                    if center_diff < self.eps:
                        break

        centers = centers + (self.cluster_centers.unsqueeze(0) -
                             self.cluster_centers.detach().unsqueeze(0))

        distances = self.compute_distances(x, centers)
        distances.masked_fill_(mask.unsqueeze(-1), 1e6)
        soft_assign = F.softmax(-distances / self.temperature, dim=-1)

        final_counts = soft_assign.sum(dim=1)
        clustered_representations = torch.bmm(soft_assign.transpose(1, 2), x)

        clustered_x = torch.zeros(bs, self.cluster_num, dim, device=device, dtype=x.dtype)
        cluster_mask = torch.zeros(bs, self.cluster_num, device=device, dtype=torch.bool)
        cluster_counts = torch.zeros(bs, self.cluster_num, device=device, dtype=x.dtype)

        clustered_x[:, :self.max_clusters, :] = clustered_representations / (final_counts.unsqueeze(-1) + self.eps)
        cluster_counts[:, :self.max_clusters] = final_counts
        cluster_mask[:, :self.max_clusters] = (final_counts > self.eps)

        has_padding = mask.any(dim=1)

        if has_padding.any():
            masked_x = x * valid_positions_mask.unsqueeze(-1).float()
            sum_valid_x = masked_x.sum(dim=1)
            num_valid_tokens__for_div = num_valid_tokens.unsqueeze(-1) + self.eps
            mean_valid_x = sum_valid_x / num_valid_tokens__for_div

            clustered_x[has_padding, self.max_clusters] = mean_valid_x[has_padding]
            cluster_mask[has_padding, self.max_clusters] = True
            cluster_counts[has_padding, self.max_clusters] = 1.0

        return clustered_x, cluster_mask, cluster_counts


def Combined_Loss(args, pred_users, labels_padded, label_mask, negative_samples, neg_mask, first_label):
    """
    Hybrid loss: BCE (multi-label) + BPR (ranking) + CE (next-user anchor).
    """
    device = pred_users.device
    batch_size, num_classes = pred_users.size()

    alpha  = getattr(args, 'alpha',      2.0)
    beta   = getattr(args, 'beta',       2.0)
    gamma  = getattr(args, 'gamma',      0.5)
    margin = getattr(args, 'bpr_margin', 1.0)

    target = torch.zeros_like(pred_users)
    valid_batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(labels_padded)[label_mask]
    valid_positives = labels_padded[label_mask]
    target[valid_batch_idx, valid_positives] = 1.0

    bce_loss = F.binary_cross_entropy_with_logits(
        pred_users[:, 1:], target[:, 1:], reduction='sum'
    ) / (batch_size * (num_classes - 1))

    pos_scores = torch.gather(pred_users, 1, labels_padded)
    neg_scores = torch.gather(pred_users, 1, negative_samples)

    combined_neg_mask = neg_mask & (negative_samples != 0)
    valid_pair_mask = label_mask.unsqueeze(2) & combined_neg_mask.unsqueeze(1)

    if valid_pair_mask.any():
        score_diff = pos_scores.unsqueeze(2) - neg_scores.unsqueeze(1)
        bpr_loss = F.softplus(-(score_diff[valid_pair_mask] - margin)).mean()
    else:
        bpr_loss = torch.tensor(0.0, device=device)

    ce_loss = F.cross_entropy(pred_users, first_label, ignore_index=0)

    return alpha * bce_loss + beta * bpr_loss + gamma * ce_loss


class WeightedPooling(nn.Module):
    def __init__(self, args):
        super(WeightedPooling, self).__init__()
        self.dim = args.dim

    def forward(self, x, cluster_mask, cluster_counts):
        masked_x = x * cluster_mask.unsqueeze(-1).float()
        weighted_x = masked_x * cluster_counts.unsqueeze(-1).float()
        summed = torch.sum(weighted_x, dim=1)
        total_counts = (cluster_counts * cluster_mask.float()).sum(dim=1, keepdim=True)
        pooled = summed / (total_counts + 1e-8)

        return pooled


class PositionEmbedding(nn.Module):
    def __init__(self, args):
        super(PositionEmbedding, self).__init__()
        self.dim = args.dim
        self.pos_embed = nn.Embedding(args.max_len + 1, self.dim)
        self.Drop = nn.Dropout(args.dropout)

    def forward(self, seq_h):
        position_ids = torch.arange(0, seq_h.size(1), dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).view(-1, seq_h.size(1))
        pe = self.pos_embed(position_ids).repeat(seq_h.size(0), 1, 1)
        return self.Drop(pe)


class SelfAttn(nn.Module):
    def __init__(self, input_size, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1):
        super(SelfAttn, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, attn_mask, epsilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + epsilon)

        pad_mask = attn_mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
        attn_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
        mask_ = attn_mask + pad_mask
        Q_K = Q_K.masked_fill(mask_, -2 ** 32 + 1)

        attn_weight = F.softmax(Q_K, dim=-1)
        attn_weight = self.dropout(attn_weight)
        return attn_weight @ V

    def multi_head_attention(self, Q, K, V, mask):

        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)
        mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V, mask):
        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class GNNConv(nn.Module):
    def __init__(self):
        super(GNNConv, self).__init__()
    
    def forward(self, x, edge_index):
        """
        LightGCN convolution: simple message passing without learnable parameters
        """
        row, col = edge_index

        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row])

        deg = torch.bincount(col, minlength=x.size(0)).float()
        deg = torch.clamp(deg, min=1.0)
        out = out / deg.unsqueeze(1)

        return out


class GNNEncoder(nn.Module):
    """
    LightGCN encoder with multiple layers and layer combination
    """
    def __init__(self, args):
        super(GNNEncoder, self).__init__()
        self.dim = args.dim
        self.n_layers = args.gnn_layers
        self.dropout = args.dropout

        self.layers = nn.ModuleList([GNNConv() for _ in range(self.n_layers)])
        self.drop = nn.Dropout(self.dropout)

        self.layer_weights = nn.Parameter(torch.ones(self.n_layers) / self.n_layers)

    def forward(self, feature, Graph):
        """
        Forward pass with layer combination
        """
        x = feature
        layer_outputs = []

        for layer in self.layers:
            x = layer(x, Graph.edge_index)
            x = self.drop(x)
            layer_outputs.append(x)

        weights = F.softmax(self.layer_weights, dim=0)
        final_output = sum(w * out for w, out in zip(weights, layer_outputs))

        return final_output


class GraphEncoder(nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        self.dim = args.dim
        self.gnn_type = getattr(args, 'gnn_type', 'gat')

        if self.gnn_type.lower() == 'lightgcn':
            self.gnn = GNNEncoder(args)
        elif self.gnn_type.lower() == 'gat':
            self.gnn = GATConv(self.dim, self.dim)
            self.drop = nn.Dropout(args.dropout)
            self.batch_norm = nn.BatchNorm1d(self.dim)
        elif self.gnn_type.lower() == 'gcn':
            self.gnn = GCNConv(self.dim, self.dim)
            self.drop = nn.Dropout(args.dropout)
            self.batch_norm = nn.BatchNorm1d(self.dim)
        elif self.gnn_type.lower() == 'sage':
            self.gnn = SAGEConv(self.dim, self.dim)
            self.drop = nn.Dropout(args.dropout)
            self.batch_norm = nn.BatchNorm1d(self.dim)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def forward(self, feature, Graph):
        if self.gnn_type.lower() == 'lightgcn':
            return self.gnn(feature, Graph)
        else:
            x = self.gnn(feature, Graph.edge_index)
            x = self.drop(x)
            x = self.batch_norm(x)
            return x