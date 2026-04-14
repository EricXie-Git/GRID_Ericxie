import torch
import torch.nn.functional as F


def group_softmax(scores, cluster_counts, dim=-1, epsilon=1e-8):
    """
    GroupSoftMax attention weighting.

    Formula: GroupSoftMax(O_ij) = count_j * exp(O_ij) / sum(count_μ * exp(O_iμ))

    Args:
        scores: Attention score tensor
        cluster_counts: Cluster size tensor (same shape as scores)
        dim: Dimension along which to normalize
        epsilon: Small value for numerical stability
    """
    max_scores = scores.max(dim=dim, keepdim=True)[0]
    scores_stable = scores - max_scores

    exp_scores = torch.exp(scores_stable)
    weighted_exp_scores = cluster_counts * exp_scores
    sum_weights = weighted_exp_scores.sum(dim=dim, keepdim=True)
    attention_weights = weighted_exp_scores / (sum_weights + epsilon)

    return attention_weights


def test_output_consistency():
    """
    Verify that GroupSoftMax attention on clustered sequences approximates
    standard attention on the original sequences across multiple cluster configurations.
    """
    print("\nTESTING GROUPSOFTMAX OUTPUT CONSISTENCY")
    print("=" * 60)

    batch_size = 4
    d_model = 8
    torch.manual_seed(42)

    scenarios = [
        {
            'name': 'Basic clustering (3->2)',
            'seq_len_orig': 3,
            'seq_len_clust': 2,
            'cluster_counts': [[2, 1], [2, 1], [2, 1], [2, 1]],
            'cluster_assignment': [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        },
        {
            'name': 'Larger clusters (6->3)',
            'seq_len_orig': 6,
            'seq_len_clust': 3,
            'cluster_counts': [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]],
            'cluster_assignment': [[0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2]]
        },
        {
            'name': 'Uneven clusters (5->3)',
            'seq_len_orig': 5,
            'seq_len_clust': 3,
            'cluster_counts': [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]],
            'cluster_assignment': [[0, 0, 1, 1, 2], [0, 0, 1, 1, 2], [0, 0, 1, 1, 2], [0, 0, 1, 1, 2]]
        }
    ]

    results = []
    for scenario in scenarios:
        print(f"\n Testing scenario: {scenario['name']}")
        print("-" * 40)

        seq_len_orig = scenario['seq_len_orig']
        seq_len_clust = scenario['seq_len_clust']

        X_original = torch.randn(batch_size, seq_len_orig, d_model)

        cluster_assignment = torch.tensor(scenario['cluster_assignment'], dtype=torch.long)

        for b in range(batch_size):
            for c in range(seq_len_clust):
                cluster_positions = (cluster_assignment[b] == c).nonzero(as_tuple=True)[0]
                if len(cluster_positions) > 1:
                    base_embedding = X_original[b, cluster_positions[0]]
                    for pos in cluster_positions[1:]:
                        X_original[b, pos] = base_embedding + 0.01 * torch.randn_like(base_embedding)

        X_clustered = torch.zeros(batch_size, seq_len_clust, d_model)
        cluster_counts = torch.tensor(scenario['cluster_counts'], dtype=torch.float)

        for b in range(batch_size):
            for c in range(seq_len_clust):
                mask = (cluster_assignment[b] == c)
                X_clustered[b, c] = X_original[b, mask].mean(dim=0)

        print(f"Embedding similarities within clusters:")
        for b in range(min(2, batch_size)):
            for c in range(seq_len_clust):
                cluster_positions = (cluster_assignment[b] == c).nonzero(as_tuple=True)[0]
                if len(cluster_positions) > 1:
                    similarities = []
                    for i in range(len(cluster_positions)):
                        for j in range(i+1, len(cluster_positions)):
                            sim = F.cosine_similarity(
                                X_original[b, cluster_positions[i]],
                                X_original[b, cluster_positions[j]],
                                dim=0
                            )
                            similarities.append(sim.item())
                    avg_sim = sum(similarities) / len(similarities)
                    print(f"  Batch {b}, Cluster {c}: avg similarity = {avg_sim:.6f}")

        Q_orig, K_orig, V_orig = X_original, X_original, X_original
        attention_scores_orig = torch.matmul(Q_orig, K_orig.transpose(-2, -1)) / (d_model ** 0.5)
        attention_weights_orig = F.softmax(attention_scores_orig, dim=-1)
        output_orig = torch.matmul(attention_weights_orig, V_orig)

        Q_clust, K_clust, V_clust = X_clustered, X_clustered, X_clustered
        attention_scores_clust = torch.matmul(Q_clust, K_clust.transpose(-2, -1)) / (d_model ** 0.5)

        expanded_counts = torch.zeros_like(attention_scores_clust)
        for b in range(batch_size):
            for i in range(seq_len_clust):
                for j in range(seq_len_clust):
                    expanded_counts[b, i, j] = cluster_counts[b, j]

        attention_weights_clust = group_softmax(attention_scores_clust, expanded_counts)
        output_clust = torch.matmul(attention_weights_clust, V_clust)

        output_clust_mapped = torch.zeros_like(output_orig)
        for b in range(batch_size):
            for i in range(seq_len_orig):
                cluster_idx = cluster_assignment[b, i]
                output_clust_mapped[b, i] = output_clust[b, cluster_idx]

        output_diff = torch.abs(output_orig - output_clust_mapped).max()

        maxpool_orig = torch.max(output_orig, dim=1)[0]
        maxpool_clust = torch.max(output_clust_mapped, dim=1)[0]
        maxpool_diff = torch.abs(maxpool_orig - maxpool_clust).max()

        meanpool_orig = torch.mean(output_orig, dim=1)
        meanpool_clust = torch.mean(output_clust_mapped, dim=1)
        meanpool_diff = torch.abs(meanpool_orig - meanpool_clust).max()

        print(f"Original sequence length: {seq_len_orig}")
        print(f"Clustered sequence length: {seq_len_clust}")
        print(f"Cluster sizes: {cluster_counts[0].tolist()}")
        print(f"Max sequence output difference: {output_diff:.8f}")
        print(f"Max maxpooled output difference: {maxpool_diff:.8f}")
        print(f"Max meanpooled output difference: {meanpool_diff:.8f}")

        results.append({
            'scenario': scenario['name'],
            'output_diff': output_diff.item(),
            'maxpool_diff': maxpool_diff.item(),
            'meanpool_diff': meanpool_diff.item()
        })

    print("\nFINAL ASSESSMENT")
    print("=" * 60)
    tolerance = 1e-2

    for result in results:
        output_success = result['output_diff'] < tolerance
        maxpool_success = result['maxpool_diff'] < tolerance
        meanpool_success = result['meanpool_diff'] < tolerance
        status = "OK" if output_success and maxpool_success and meanpool_success else "NG"
        print(f"{status} {result['scenario']}:")
        print(f"    Sequence diff = {result['output_diff']:.8f}")
        print(f"    Maxpool diff = {result['maxpool_diff']:.8f}")
        print(f"    Meanpool diff = {result['meanpool_diff']:.8f}")

    all_success = all(r['output_diff'] < tolerance and r['maxpool_diff'] < tolerance and r['meanpool_diff'] < tolerance for r in results)
    if all_success:
        print(f"\n [SUCCESS] GroupSoftMax produces consistent outputs across all scenarios!")
        print(f"   - Sequence-level, maxpooled, and meanpooled outputs all match within tolerance ({tolerance})")
        print(f"   - Small differences are due to numerical precision")
    else:
        print(f"\n [ISSUE] GroupSoftMax outputs show significant differences in some scenarios")
        print(f"   - Differences > {tolerance} suggest potential problems")


if __name__ == "__main__":
    test_output_consistency()
