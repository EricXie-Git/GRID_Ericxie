import torch


class Metrics(object):
    """Evaluation metrics for cascade prediction: Recall@K and NDCG@K."""

    def __init__(self, args):
        super().__init__()
        self.PAD = 0
        self.k_list = args.metric_k

    def compute_recall_ndcg_multi_label(self, y_prob, labels_padded):
        device = y_prob.device
        labels_padded = labels_padded.to(device)

        batch_size = y_prob.size(0)
        if not self.k_list or batch_size == 0:
            return {}, batch_size

        max_k = min(max(self.k_list), y_prob.size(1))
        if max_k == 0:
            return {}, batch_size

        _, top_k_indices = torch.topk(y_prob, k=max_k, dim=1)

        valid_mask = labels_padded != self.PAD
        num_true = valid_mask.sum(dim=1)
        num_true_clamped = num_true.clamp(min=1).float()

        matches = (
            (top_k_indices.unsqueeze(2) == labels_padded.unsqueeze(1))
            & valid_mask.unsqueeze(1)
        )
        match_any = matches.any(dim=2).float()

        log_denom = torch.log2(
            torch.arange(2, max_k + 2, device=device, dtype=torch.float32)
        )

        positions = torch.arange(max_k, device=device)
        ideal_match = (positions < num_true.unsqueeze(1)).float()

        scores = {}
        for k in self.k_list:
            k = min(k, max_k)
            ld        = log_denom[:k]
            rel       = match_any[:, :k]
            ideal_rel = ideal_match[:, :k]

            num_hits = rel.sum(dim=1)
            scores[f'recall@{k}'] = (num_hits / num_true_clamped).sum()

            dcg  = (rel       / ld).sum(dim=1)
            idcg = (ideal_rel / ld).sum(dim=1)
            scores[f'ndcg@{k}'] = (dcg / idcg.clamp(min=1e-8)).sum()

        return scores, batch_size
