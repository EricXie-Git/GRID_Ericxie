"""分析社交图的层级性、树状程度和度分布尾部，默认分析 Weibo。

用法：
    python data_analyse.py --dataset weibo
    python data_analyse.py --dataset twitter
    python data_analyse.py --dataset douban --no-plots
    python data_analyse.py --dataset weibo --graph-path dataset/weibo/graph.pt
    python data_analyse.py --dataset weibo --bootstrap 1000 --overwrite

基础依赖：numpy scipy networkx；绘图需 matplotlib；读取 PT 另需 torch torch-geometric。
默认优先读取 origin_data（也兼容 origin）中的 graph.txt/graph.npz，再寻找 graph.pt。
NPZ 默认选择关系位 1，与本项目预处理保持一致；TXT/PT 不进行关系位筛选。
Twitter 可直接读取 dataset/origin_data/twitter/graph.npz，无需先进行数据预处理。
其真实用户 ID 为 1..12627（来自 config.py），默认输出到 analysis/twitter。
也可用 --dataset twitter --graph-path dataset/twitter/graph.pt 分析训练用图。
分析前去掉 PAD、末尾特殊节点、自环和重复边，所有指标按无权简单图计算。
auto 模式将完全双向对称的边集合视为无向图，这只是存储约定的推断，可手动覆盖。

输出 analysis.json（全部指标）、summary.md（中文说明）和 diagnostics.png（可选）。
另输出 degree_distribution_linear.png / degree_distribution_loglog.png，分别用普通/双对数坐标绘制度分布。
默认随机四点法计算平均 δ_avg 和归一化 δ_G=2δ_avg/d_avg；--hyperbolicity off 可跳过。
非连通图默认分析最大连通分量并报告覆盖率；平均值不是最大 Gromov δ。
层级指标不是因果传播层次；BFS 层数不是原图为树的证据。
幂律使用离散极大似然和 KS 选择尾部阈值，不用双对数直线回归证明幂律。
默认运行 1000 次 bootstrap；可用 --bootstrap 0 跳过，跳过时不判断是否服从幂律。
即使 bootstrap 不拒绝，也不证明幂律优于所有其他分布，且网络度数并非独立样本。
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parent
HYPERBOLICITY_METHOD_REFERENCE = "方法参照：[1]谢文锦.社交与学术网络上的信息传播预测研究[D].西南大学,2025.DOI:10.27684/d.cnki.gxndx.2025.003879."
REFERENCES = {
    "power_law": "https://arxiv.org/abs/0706.1062",
    "clustering_hierarchy": "https://arxiv.org/abs/cond-mat/0206130",
    "hyperbolicity": "https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hyperbolicity.html",
}
PLOT_FILES = ("diagnostics.png", "degree_distribution_linear.png", "degree_distribution_loglog.png")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="weibo",
                        help="数据集名称，如 twitter、douban、weibo；默认 weibo，也支持自定义目录名")
    parser.add_argument("--graph-path", type=Path, help="显式指定 .txt/.npz/.pt 图文件")
    parser.add_argument("--output-dir", type=Path, help="默认 analysis/<dataset>")
    parser.add_argument("--max-user-id", type=int, help="真实用户编号上限；0 始终视为 PAD")
    parser.add_argument("--relation-bit", type=int, default=1, help="仅对 NPZ 有效，默认 1")
    parser.add_argument("--direction", choices=("auto", "directed", "undirected"), default="auto")
    parser.add_argument("--clustering-samples", type=int, default=1000, help="聚类系数抽样节点数，0 表示全部")
    parser.add_argument("--min-tail", type=int, default=50, help="幂律拟合的最少尾部节点数")
    parser.add_argument("--max-xmin-candidates", type=int, default=64, help="搜索阈值数量上限，0 表示全部")
    parser.add_argument("--bootstrap", type=int, default=1000, help="半参数模拟次数，默认 1000，每次重新拟合；0 表示跳过")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--hyperbolicity", choices=("sampled", "off"), default="sampled",
                        help="随机四点平均双曲度；off 跳过")
    parser.add_argument("--hyperbolicity-samples", type=int, default=100000,
                        help="全分量均匀抽样的互异四元组数量，默认 100000")
    parser.add_argument("--hyperbolicity-pairs", type=int, default=100000,
                        help="用于估计 d_avg 的独立互异节点对数量，默认 100000")
    parser.add_argument("--hyperbolicity-scope", choices=("largest", "require-connected"), default="largest",
                        help="默认分析最大连通分量；require-connected 在非连通图上报错")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.dataset or Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
        parser.error("dataset 必须是单个目录名")
    args.dataset = args.dataset.lower()
    if args.max_user_id is not None and args.max_user_id < 1:
        parser.error("max-user-id 必须为正数")
    if args.relation_bit < 1 or args.relation_bit & (args.relation_bit - 1):
        parser.error("relation-bit 必须为 1、2、4 等单个关系位")
    if min(args.clustering_samples, args.bootstrap, args.max_xmin_candidates) < 0 or args.min_tail < 2:
        parser.error("采样/模拟次数不能为负，min-tail 至少为 2")
    args.output_dir = (args.output_dir or ROOT / "analysis" / args.dataset).resolve()
    if args.hyperbolicity_samples < 1 or args.hyperbolicity_pairs < 1:
        parser.error("双曲度 samples/pairs 必须为正整数")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or not args.overwrite):
        parser.error("输出目录已存在，请指定新目录或使用 --overwrite")
    return args


def find_graph(args):
    if args.graph_path:
        path = args.graph_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for root in (ROOT / "dataset/origin_data", ROOT / "dataset/origin", ROOT / "dataset"):
        if not root.is_dir():
            continue
        matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower() == args.dataset)
        for folder in matches:
            for name in ("graph.txt", "graph.npz", "graph.pt"):
                if (folder / name).is_file():
                    return folder / name
    raise FileNotFoundError("没有找到图文件，请使用 --graph-path")


def load_graph(args):
    """读取边和完整用户编号范围，不能仅统计有边节点而遗漏孤立用户。"""
    import numpy as np
    import networkx as nx
    from config import DATASET_CONFIGS

    path = find_graph(args)
    preset = DATASET_CONFIGS.get(args.dataset)
    # Twitter/Douban/Weibo 共用读取和分析流程；配置给出真实用户范围，保留孤立用户。
    max_id = args.max_user_id if args.max_user_id is not None else (preset.user_num - 1 if preset else None)
    id_source = "argument" if args.max_user_id is not None else "config.py" if preset else "inferred"
    suffix = path.suffix.lower()
    if suffix != ".npz" and args.relation_bit != 1:
        raise ValueError("TXT/PT 不使用关系位，请移除 --relation-bit")
    if suffix == ".txt":
        edges = []
        with path.open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 2:
                    raise ValueError(f"{path.name}:{line_no} 应为两列整数用户 ID")
                try:
                    edges.append(tuple(map(int, fields)))
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{line_no} 用户 ID 不是整数") from exc
        edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if max_id is None:
            max_id = int(edge_array.max()) if edge_array.size else 0
            id_source = "maximum endpoint; assumes contiguous IDs, unknown isolated users cannot be recovered"
    elif suffix == ".npz":
        import scipy.sparse as sparse
        adj = sparse.load_npz(path)
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("NPZ 图必须为方阵")
        coo = adj.tocoo()
        values = coo.data
        if (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
                or not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**63).any() or not (values == np.floor(values)).all()):
            raise ValueError("NPZ 关系编码必须为非负整数")
        keep = (values.astype(np.int64) & args.relation_bit) != 0
        edge_array = np.column_stack((coo.row[keep], coo.col[keep])).astype(np.int64)
        if max_id is None:
            max_id = adj.shape[0] - 2
            id_source = "NPZ size minus 2; assumes PAD and trailing special node"
        if max_id >= adj.shape[0]:
            raise ValueError("用户范围超出 NPZ 维度")
    elif suffix == ".pt":
        # PT 包含 Python 序列化对象，仅加载自己生成或信任来源的文件。
        import torch
        import torch_geometric  # noqa: F401
        graph = torch.load(path, map_location="cpu", weights_only=False)
        edge_index = graph.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.is_floating_point():
            raise ValueError("PT edge_index 应为整数张量 [2, E]")
        edge_array = edge_index.cpu().numpy().T.astype(np.int64)
        if max_id is None:
            max_id = int(graph.num_nodes) - 1
            id_source = "PT num_nodes minus PAD; assumes no trailing special node"
        if max_id >= graph.num_nodes:
            raise ValueError("用户范围超出 PT 节点数")
    else:
        raise ValueError("仅支持 .txt、.npz、.pt 图文件")
    if max_id < 1:
        raise ValueError("无法确定真实用户范围，请提供 --max-user-id")
    if edge_array.size and (edge_array < 0).any():
        raise ValueError("图包含负数用户 ID")
    valid = ((edge_array >= 1) & (edge_array <= max_id)).all(axis=1)
    loops = edge_array[:, 0] == edge_array[:, 1]
    # NPZ 中特殊节点自环可排除，但真实关系连向范围外节点通常意味着配置错误。
    if np.any(~valid & ~loops):
        raise ValueError("有非自环边超出用户范围，请核对 --max-user-id")
    selected = edge_array[valid & ~loops]
    edges = np.unique(selected, axis=0)
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(1, max_id + 1))
    digraph.add_edges_from(edges.tolist())
    symmetric = all(digraph.has_edge(v, u) for u, v in digraph.edges)
    directed = args.direction == "directed" or (args.direction == "auto" and not symmetric)
    graph = digraph if directed else digraph.to_undirected()
    meta = {"path": str(path), "format": suffix, "user_id_source": id_source,
            "max_user_id": max_id, "selected_raw_edge_records": len(edge_array),
            "special_node_records_removed": int((~valid).sum()),
            "real_user_self_loop_records_removed": int((loops & valid).sum()),
            "duplicate_nonloop_records_removed": len(selected) - len(edges),
            "relation_bit": args.relation_bit if suffix == ".npz" else None,
            "all_edges_reciprocal": symmetric, "direction_option": args.direction,
            "analysed_as_directed": directed, "weights_used": False}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    meta["sha256"] = digest.hexdigest()
    return graph, meta


def histogram(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def structure_metrics(graph, sample_size, seed):
    """树状性用无向投影；有向层次单独用 SCC 压缩，避免把互惠边当作两条无向边。"""
    import networkx as nx
    import numpy as np

    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=lambda c: (-len(c), min(c)))
    n, m = undirected.number_of_nodes(), undirected.number_of_edges()
    degrees = dict(undirected.degree())
    cycle_rank = m - n + len(components)
    tree_sizes = []
    for nodes in components:
        component_edges = sum(degrees[u] for u in nodes) // 2
        if component_edges == len(nodes) - 1:
            tree_sizes.append(len(nodes))
    bridge_count = sum(1 for _ in nx.bridges(undirected))
    tree = {"projection": "simple undirected, no loops", "nodes": n, "edges": m,
            "components": len(components), "isolates": sum(d == 0 for d in degrees.values()),
            "leaves": sum(d == 1 for d in degrees.values()),
            "leaf_fraction_nonisolated": sum(d == 1 for d in degrees.values()) / max(1, sum(d > 0 for d in degrees.values())),
            "is_tree": len(components) == 1 and cycle_rank == 0,
            "is_forest": cycle_rank == 0, "independent_cycles": cycle_rank,
            "min_edge_removal_fraction_to_forest": cycle_rank / m if m else 0.0,
            "bridges": bridge_count, "bridge_fraction": bridge_count / m if m else None,
            "tree_components_including_singletons": len(tree_sizes),
            "tree_components_excluding_singletons": sum(s > 1 for s in tree_sizes),
            "nodes_in_nontrivial_tree_components": sum(s for s in tree_sizes if s > 1),
            "largest_component_nodes": len(components[0]), "component_size_histogram": histogram(map(len, components))}
    # k-core 描述核心/外围层次，不等于传播时间顺序。
    cores = nx.core_number(undirected)
    hierarchy = {"max_core": max(cores.values()), "core_histogram": histogram(cores.values())}
    giant = undirected.subgraph(components[0])
    root = min(giant, key=lambda u: (-degrees[u], u))
    distances = nx.single_source_shortest_path_length(giant, root)
    hierarchy["bfs_largest_component"] = {
        "root": root, "root_rule": "highest degree, smallest ID breaks ties",
        "max_distance": max(distances.values()), "level_sizes": histogram(distances.values()),
        "note": "Distances from one selected root, NOT tree depth or graph diameter."}
    nodes = sorted(undirected.nodes)
    rng = np.random.default_rng(seed)
    sampled = nodes if sample_size == 0 or sample_size >= n else sorted(rng.choice(nodes, sample_size, replace=False).tolist())
    clustering = nx.clustering(undirected, sampled)
    grouped = defaultdict(list)
    for user, value in clustering.items():
        grouped[degrees[user]].append(value)
    ck = [{"degree": degree, "mean_clustering": float(np.mean(vals)), "sample_nodes": len(vals)}
          for degree, vals in sorted(grouped.items())]
    hierarchy["clustering"] = {"sampled_nodes": len(sampled), "population_nodes": n,
                                "mean": float(np.mean(list(clustering.values()))), "by_degree": ck}
    # C(k) 的双对数斜率只描述趋势，不将其当作层级存在性的统计检验。
    usable = [r for r in ck if r["degree"] >= 2 and r["mean_clustering"] > 0]
    if len(usable) >= 3:
        x = np.log([r["degree"] for r in usable]); y = np.log([r["mean_clustering"] for r in usable])
        slope, intercept = np.polyfit(x, y, 1)
        total = float(np.sum((y - y.mean()) ** 2))
        hierarchy["clustering"]["loglog_trend"] = {
            "slope": float(slope), "r_squared": float(1 - np.sum((y - (slope*x + intercept))**2)/total) if total else None,
            "degree_bins": len(usable), "note": "Unweighted descriptive fit of positive C(k) means; not proof of hierarchy."}
    if graph.is_directed():
        sccs = list(nx.strongly_connected_components(graph))
        dag = nx.condensation(graph, sccs)
        levels = {}
        for u in nx.topological_sort(dag):
            levels[u] = max((levels[v] + 1 for v in dag.predecessors(u)), default=0)
        weighted_levels = Counter()
        for u, level in levels.items():
            weighted_levels[level] += len(dag.nodes[u]["members"])
        inside = sum(dag.graph["mapping"][u] == dag.graph["mapping"][v] for u, v in graph.edges)
        hierarchy["directed"] = {
            "is_dag_after_loop_removal": all(len(c) == 1 for c in sccs),
            "strong_components": len(sccs), "largest_scc_nodes": max(map(len, sccs)),
            "largest_scc_fraction": max(map(len, sccs)) / n,
            "within_scc_edge_fraction": inside / graph.number_of_edges() if graph.number_of_edges() else None,
            "condensation_edges": dag.number_of_edges(), "condensation_longest_path_edges": max(levels.values()),
            "condensation_level_node_counts": dict(sorted(weighted_levels.items())),
            "reciprocity": nx.reciprocity(graph) if graph.number_of_edges() else None,
            "source_nodes": sum(graph.in_degree(u) == 0 and graph.out_degree(u) > 0 for u in graph),
            "sink_nodes": sum(graph.out_degree(u) == 0 and graph.in_degree(u) > 0 for u in graph),
            "out_arborescence_components_nontrivial": sum(
                len(c) > 1 and nx.is_arborescence(graph.subgraph(c)) for c in components),
            "note": "SCC condensation is always a DAG; its levels alone do not prove the original graph is hierarchical."}
    return tree, hierarchy


def random_distinct_rows(rng, n, count, width):
    """均匀抽取互异节点；不同样本可以重复，不限制候选节点池。"""
    import numpy as np
    rows = rng.integers(0, n, size=(count, width))
    while True:
        invalid = (np.diff(np.sort(rows, axis=1), axis=1) == 0).any(axis=1)
        if not invalid.any():
            return rows
        rows[invalid] = rng.integers(0, n, size=(int(invalid.sum()), width))


def queried_distances(graph, ids, pairs):
    """对需要的节点对求精确最短路，分批 BFS，避免存储全源 n*n 距离矩阵。"""
    import numpy as np
    import networkx as nx
    from scipy.sparse.csgraph import shortest_path
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=ids, dtype=np.float64, format="csr")
    adjacency.indices = adjacency.indices.astype(np.int32)
    adjacency.indptr = adjacency.indptr.astype(np.int32)
    # 无向距离对称，统一端点次序减少重复 BFS；结果恢复原查询顺序。
    pairs = np.sort(pairs, axis=1)
    order = np.argsort(pairs[:, 0], kind="stable")
    sorted_pairs = pairs[order]
    sources, starts = np.unique(sorted_pairs[:, 0], return_index=True)
    ends = np.r_[starts[1:], len(pairs)]
    output = np.empty(len(pairs), dtype=np.float64)
    for begin in range(0, len(sources), 16):
        selected = sources[begin:begin+16]
        block = shortest_path(adjacency, directed=False, unweighted=True, indices=selected)
        for row in range(len(selected)):
            i = begin + row
            positions = slice(starts[i], ends[i])
            output[order[positions]] = block[row, sorted_pairs[positions, 1]]
        if begin % 2048 == 0 or begin + 16 >= len(sources):
            print(f"  Shortest paths: {min(begin+16, len(sources))}/{len(sources)} sources", flush=True)
    if not np.isfinite(output).all():
        raise ValueError("平均双曲度分析范围内存在不可达节点对")
    return output


def hyperbolicity_metrics(graph, args):
    """按用户给定公式计算平均四点偏差及 2*delta_avg/d_avg，不计算全局最大值。

    方法参照：[1]谢文锦.社交与学术网络上的信息传播预测研究[D].西南大学,2025.DOI:10.27684/d.cnki.gxndx.2025.003879.

    四元组在完整分析分量上均匀抽样，组内四节点互异；d_avg 使用独立抽样
    的互异节点对。非连通图默认采用最大连通分量，并显式报告覆盖范围。
    归一化比值不截断：该公式并不能保证结果落在 [0,1]（四节点环约为 1.5）。
    """
    import networkx as nx
    import numpy as np
    result = {"method": "uniform_quadruple_mean", "mode": args.hyperbolicity,
              "method_reference": HYPERBOLICITY_METHOD_REFERENCE,
              "definition": "delta_avg=mean((largest_pair_sum-second_largest_pair_sum)/2)",
              "normalization": "delta_G=2*delta_avg/d_avg; not clipped to [0,1]",
              "metric": "unweighted undirected vertex shortest-path metric", "seed": args.seed,
              "quadruples_requested": args.hyperbolicity_samples,
              "pairs_requested": args.hyperbolicity_pairs, "scope": args.hyperbolicity_scope}
    if args.hyperbolicity == "off":
        return {**result, "status": "skipped"}
    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=lambda c: (-len(c), min(c)))
    if args.hyperbolicity_scope == "require-connected" and len(components) != 1:
        raise ValueError("平均双曲度要求连通图；可使用 --hyperbolicity-scope largest 分析最大连通分量")
    ids = sorted(components[0]) if components else []
    n = len(ids)
    result.update({"total_nodes": len(undirected), "components": len(components),
                   "analysed_nodes": n, "excluded_nodes": len(undirected)-n,
                   "node_coverage": n/len(undirected) if len(undirected) else 0.0,
                   "delta_avg": None, "d_avg": None, "delta_G": None,
                   "quadruples_evaluated": 0, "pairs_evaluated": 0})
    # 少于四个节点无法按指定的互异四元组定义估计，不伪造零值。
    if n < 4:
        return {**result, "status": "insufficient_nodes"}
    rng_quad, rng_pair = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2)]
    quads = random_distinct_rows(rng_quad, n, args.hyperbolicity_samples, 4)
    pairs = random_distinct_rows(rng_pair, n, args.hyperbolicity_pairs, 2)
    # 距离顺序为 ab, cd, ac, bd, ad, bc；随后为独立节点对距离。
    quad_pairs = quads[:, [0,1,2,3,0,2,1,3,0,3,1,2]].reshape(-1, 2)
    queries = np.concatenate((quad_pairs, pairs))
    print(f"Mean hyperbolicity: {n} nodes, {len(quads)} quadruples, {len(pairs)} pairs", flush=True)
    distances = queried_distances(undirected.subgraph(ids), ids, queries)
    sums = distances[:6*len(quads)].reshape(-1, 3, 2).sum(axis=2)
    sums.sort(axis=1)
    delta = (sums[:, 2]-sums[:, 1])/2
    pair_distances = distances[6*len(quads):]
    delta_avg, d_avg = float(delta.mean()), float(pair_distances.mean())
    # 标准误仅反映固定图上随机抽样的误差，不是全局最大 δ 的置信区间。
    def mean_se(values):
        return float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None
    delta_se, distance_se = mean_se(delta), mean_se(pair_distances)
    ratio_se = (math.sqrt((2*delta_se/d_avg)**2 + (2*delta_avg*distance_se/d_avg**2)**2)
                if delta_se is not None and distance_se is not None else None)
    result.update({"status": "sampled_mean", "quadruples_evaluated": len(quads), "pairs_evaluated": len(pairs),
                   "delta_avg": delta_avg, "d_avg": d_avg, "delta_G": 2*delta_avg/d_avg,
                   "delta_avg_mc_se": delta_se, "d_avg_mc_se": distance_se,
                   "delta_G_approx_mc_se": ratio_se,
                   "sampled_delta_histogram": {str(float(v)): int(c) for v,c in zip(*np.unique(delta, return_counts=True))},
                   "pair_distance_histogram": {str(int(v)): int(c) for v,c in zip(*np.unique(pair_distances, return_counts=True))},
                   "note": "Mean statistic, not maximum Gromov delta. Small values alone do not establish a tree or hierarchy."})
    return result


def fit_discrete_powerlaw(values, min_tail=50, max_candidates=64):
    """离散模型 P(k)=k^-alpha / zeta(alpha,xmin)；MLE 拟合，最小 KS 选 xmin。"""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import zeta

    values = np.sort(np.asarray(values, dtype=np.int64))
    values = values[values > 0]
    unique, counts = np.unique(values, return_counts=True)
    sizes = np.cumsum(counts[::-1])[::-1]
    candidates = unique[(sizes >= min_tail) & (unique < (unique[-1] if len(unique) else 0))]
    all_candidates = len(candidates)
    if max_candidates and all_candidates > max_candidates:
        candidates = candidates[np.unique(np.linspace(0, all_candidates-1, max_candidates).astype(int))]
    best = None
    for xmin in candidates:
        tail = values[values >= xmin]
        log_sum = float(np.log(tail).sum())
        result = minimize_scalar(lambda a: a*log_sum + len(tail)*np.log(zeta(a, float(xmin))),
                                 bounds=(1.0001, 20.0), method="bounded")
        if not result.success or not np.isfinite(result.fun):
            continue
        alpha = float(result.x)
        ks_values, ks_counts = np.unique(tail, return_counts=True)
        after = np.cumsum(ks_counts)/len(tail); before = after - ks_counts/len(tail)
        normalizer = zeta(alpha, float(xmin))
        # 同时比较跳跃前、后的 CDF，避免离散重复值造成 KS 偏差。
        ks = float(max(np.max(np.abs(before-(1-zeta(alpha, ks_values.astype(float))/normalizer))),
                       np.max(np.abs(after-(1-zeta(alpha, ks_values.astype(float)+1)/normalizer)))))
        if best is None or ks < best["ks"]:
            best = {"xmin": int(xmin), "alpha": alpha, "ks": ks, "tail_nodes": len(tail),
                    "tail_fraction_positive": len(tail)/len(values), "tail_max": int(tail[-1]),
                    "tail_decades": float(np.log10(tail[-1]/xmin)),
                    "alpha_near_search_bound": alpha < 1.001 or alpha > 19.99}
    if best:
        best.update({"searched_xmin_candidates": len(candidates), "available_xmin_candidates": all_candidates})
    return best


def sample_discrete_tail(rng, size, alpha, xmin):
    """用离散 CDF 二分反演采样，避免对高 xmin 做低效率的 Zipf 拒绝采样。"""
    import numpy as np
    from scipy.special import zeta
    targets = rng.random(size)
    lo = np.full(size, xmin-1, dtype=np.int64)
    hi = np.full(size, xmin, dtype=np.int64)
    norm = zeta(alpha, float(xmin))
    for _ in range(52):
        grow = 1-zeta(alpha, hi.astype(float)+1)/norm < targets
        if not grow.any():
            break
        if np.any(hi[grow] > 2**51):
            raise ValueError("Bootstrap tail exceeds numerical sampling range; increase xmin or skip bootstrap")
        hi[grow] *= 2
    else:
        raise ValueError("Bootstrap inverse CDF failed to bracket tail")
    while np.any(hi-lo > 1):
        mid = (hi+lo)//2
        right = 1-zeta(alpha, mid.astype(float)+1)/norm < targets
        lo = np.where(right, mid, lo); hi = np.where(right, hi, mid)
    return hi


def degree_analysis(values, args, seed):
    import numpy as np
    from scipy.special import zeta
    values = np.asarray(values, dtype=np.int64)
    positive = values[values > 0]
    unique, counts = np.unique(positive, return_counts=True)
    sorted_values = np.sort(values)
    total = int(values.sum())
    result = {"nodes": len(values), "zeros": int((values == 0).sum()),
              "degree_histogram": histogram(values),
              "mean": float(values.mean()), "max": int(values.max()),
              "gini": float(2*np.dot(np.arange(1,len(values)+1),sorted_values)/(len(values)*total)-(len(values)+1)/len(values)) if total else 0.0,
              "top_1_percent_degree_share": float(sorted_values[-max(1,math.ceil(.01*len(values))):].sum()/total) if total else None,
              "positive_degree_ccdf": [{"degree": int(k), "survival": float(p)} for k,p in
                  zip(unique, np.cumsum(counts[::-1])[::-1]/max(1,len(positive)))]}
    fit = fit_discrete_powerlaw(positive, args.min_tail, args.max_xmin_candidates)
    result["power_law"] = fit
    if fit is None:
        result["status"] = "insufficient nonconstant tail for fitting"
        return result
    tail = positive[positive >= fit["xmin"]]
    # 同一尾部上比较离散幂律与移位几何分布（离散指数）。正的 R 偏向幂律。
    q = 1/(float(tail.mean())-fit["xmin"]+1)
    log_pl = -fit["alpha"]*np.log(tail)-np.log(zeta(fit["alpha"], float(fit["xmin"])))
    log_exp = np.log(q)+(tail-fit["xmin"])*np.log1p(-q)
    diff = log_pl-log_exp
    sd = float(diff.std())
    fit["vs_discrete_exponential"] = {
        "log_likelihood_ratio": float(diff.sum()),
        "approx_two_sided_p": math.erfc(abs(float(diff.sum()))/(math.sqrt(2*len(tail))*sd)) if sd > 0 else None,
        "note": "Same fitted tail; approximate iid likelihood-ratio comparison, not a goodness-of-fit test."}
    fit["bootstrap"] = {"requested": args.bootstrap, "p_value": None}
    if args.bootstrap:
        rng = np.random.default_rng(seed)
        body = positive[positive < fit["xmin"]]
        ks_samples = []
        for iteration in range(args.bootstrap):
            tail_size = int(rng.binomial(len(positive),len(tail)/len(positive)))
            simulated = sample_discrete_tail(rng, tail_size, fit["alpha"], fit["xmin"])
            if tail_size < len(positive):
                simulated = np.concatenate((rng.choice(body,len(positive)-tail_size,replace=True),simulated))
            refit = fit_discrete_powerlaw(simulated,args.min_tail,args.max_xmin_candidates)
            if refit is None:
                fit["bootstrap"]["error"] = "A simulated sample could not be refitted; no p-value reported."
                break
            ks_samples.append(refit["ks"])
            if (iteration + 1) % 100 == 0 or iteration + 1 == args.bootstrap:
                print(f"  Bootstrap {iteration + 1}/{args.bootstrap}", flush=True)
        if len(ks_samples) == args.bootstrap:
            p = (1+sum(d >= fit["ks"] for d in ks_samples))/(args.bootstrap+1)
            fit["bootstrap"].update({"p_value": p, "completed": len(ks_samples),
                                     "monte_carlo_se": math.sqrt(p*(1-p)/(args.bootstrap+1))})
    p = fit["bootstrap"]["p_value"]
    result["status"] = ("fit only; goodness of fit not established" if p is None else
                         "power-law tail rejected at 0.1 threshold" if p < .1 else
                         "power-law tail not rejected; not proof of a power law")
    return result


def write_plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import zeta

    fig, axes = plt.subplots(2,2,figsize=(12,9), constrained_layout=True)
    for name, stats in report["degree_distributions"].items():
        rows=stats["positive_degree_ccdf"]
        if not rows:
            continue
        line,=axes[0,0].loglog([r["degree"] for r in rows],[r["survival"] for r in rows],label=name)
        fit=stats["power_law"]
        if fit:
            x=np.unique(np.geomspace(fit["xmin"],fit["tail_max"],100).astype(int))
            axes[0,0].loglog(x,fit["tail_fraction_positive"]*zeta(fit["alpha"],x.astype(float))/zeta(fit["alpha"],float(fit["xmin"])),
                            '--',color=line.get_color(),alpha=.8)
    axes[0,0].set(title="Positive-degree CCDF (dashed: fitted tail)",xlabel="Degree k",ylabel="P(K >= k | K > 0)")
    if axes[0,0].lines:
        axes[0,0].legend()
    rows=[r for r in report["hierarchy"]["clustering"]["by_degree"] if r["degree"]>=2 and r["mean_clustering"]>0]
    if rows:
        axes[0,1].loglog([r["degree"] for r in rows],[r["mean_clustering"] for r in rows],'.',alpha=.6)
    axes[0,1].set(title="Clustering by degree (positive means only)",xlabel="Undirected degree",ylabel="Mean C(k)")
    core=report["hierarchy"]["core_histogram"]
    axes[1,0].bar([int(k) for k in core],list(core.values()))
    axes[1,0].set(title="Core decomposition",xlabel="Core number",ylabel="Nodes")
    bfs=report["hierarchy"]["bfs_largest_component"]
    axes[1,1].bar([int(k) for k in bfs["level_sizes"]],list(bfs["level_sizes"].values()))
    axes[1,1].set(title=f"Largest-component BFS from node {bfs['root']}",xlabel="Distance (not causal depth)",ylabel="Nodes")
    fig.suptitle(f"{report['dataset']} - social graph diagnostics")
    fig.savefig(path,dpi=160)
    plt.close(fig)


def write_degree_plots(report, output_dir):
    """绘制完整经验度分布 P(K=k)，不分箱、不拟合，与原有的 CCDF 区分。

    两种坐标均用全部节点数作分母；对数图仅隐藏 k=0，不重新归一化。
    有向图分别画无向投影度、入度和出度，避免多条分布重叠难以辨认。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distributions = report["degree_distributions"]
    for log_scale, filename in ((False, PLOT_FILES[1]), (True, PLOT_FILES[2])):
        fig, axes = plt.subplots(1, len(distributions), figsize=(5 * len(distributions), 4),
                                 squeeze=False, constrained_layout=True)
        for ax, (name, stats) in zip(axes[0], distributions.items()):
            # 只画实际出现的度数；未出现的度数概率为零，在对数坐标上无定义。
            rows = sorted((int(k), count) for k, count in stats["degree_histogram"].items()
                          if count > 0 and (not log_scale or int(k) > 0))
            ax.scatter([k for k, _ in rows], [count / stats["nodes"] for _, count in rows],
                       s=12, alpha=0.7)
            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")
                if not rows:
                    ax.set_xlim(0.8, 2)
                    ax.set_ylim(0.1, 1)
                    ax.text(0.5, 0.5, "No positive-degree nodes", transform=ax.transAxes, ha="center")
            else:
                ax.set_xlim(left=-0.5)
                ax.set_ylim(bottom=0)
            zero_fraction = stats["zeros"] / stats["nodes"]
            ax.set(title=f"{name} | N={stats['nodes']:,}\nP(K=0)={zero_fraction:.4%}"
                         + (" (omitted on log axes)" if log_scale else ""),
                   xlabel="Degree k", ylabel="P(K = k) = count / all nodes")
            ax.grid(True, alpha=0.2)
        scale = "log-log axes; k > 0" if log_scale else "linear axes; includes k = 0"
        fig.suptitle(f"{report['dataset']} - empirical degree distribution\n{scale}", fontsize=11)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def summary_text(report):
    tree=report["tree_structure"]; hierarchy=report["hierarchy"]
    lines=[f"# {report['dataset']} 图结构分析", "", f"输入：`{report['input']['path']}`。",
           "分析对象是去掉特殊节点、自环、重复边后的无权社交图，不是传播树。",
           f"方向处理：{'有向' if report['input']['analysed_as_directed'] else '无向'}；模式 {report['input']['direction_option']}。", "",
           "## 树状结构（无向投影）", "",
           f"- 节点 {tree['nodes']}，边 {tree['edges']}，连通分量 {tree['components']}，孤立节点 {tree['isolates']}。",
           f"- 是否为树：{tree['is_tree']}；是否为森林：{tree['is_forest']}。",
           f"- 独立环数量 $m-n+c$：{tree['independent_cycles']}；变为森林至少需移除 {tree['min_edge_removal_fraction_to_forest']:.2%} 的边。",
           f"- 桥 {tree['bridges']} 条；叶节点 {tree['leaves']} 个；非孤立树组件 {tree['tree_components_excluding_singletons']} 个。",
           "环冗余越低、桥占比越高，通常越接近森林，但没有统一的“树状”分类阈值。", "",
           "## 层级特征", "", f"- 最大核数：{hierarchy['max_core']}（核心/外围结构指标）。",
           f"- 最大连通分量中，从最高度节点出发的最远距离：{hierarchy['bfs_largest_component']['max_distance']}。",
           f"- 聚类系数使用 {hierarchy['clustering']['sampled_nodes']} 个节点；$C(k)$ 趋势仅作描述。"]
    if "directed" in hierarchy:
        d=hierarchy["directed"]
        lines.extend([f"- 最大强连通分量覆盖 {d['largest_scc_fraction']:.2%} 的节点；压缩图最长路径为 {d['condensation_longest_path_edges']} 条边。",
                      "强连通分量内部存在可互达结构；压缩图必为 DAG，不能据此宣称原图具有严格层级。"])
    lines.append("BFS 层次依赖根节点；这些结构指标不证明传播方向或因果关系。")
    if "hyperbolicity" in report:
        h = report["hyperbolicity"]
        lines.extend(["", r"## 平均 $\delta$-双曲度（随机四点法）", "",
                      h.get("method_reference", HYPERBOLICITY_METHOD_REFERENCE), ""])
        if h["status"] == "skipped":
            lines.append("本次跳过平均双曲度分析。")
        elif h["status"] == "insufficient_nodes":
            lines.append("分析分量不足四个节点，无法抽取四个互异节点；结果记为 null。")
        else:
            lines.extend(["按给定公式计算四点偏差的均值，不是所有四点偏差的最大值。",
                          f"- 范围：无向投影的最大连通分量，覆盖 {h['analysed_nodes']}/{h['total_nodes']} 个节点（{h['node_coverage']:.2%}），排除 {h['excluded_nodes']} 个节点。",
                          f"- 随机互异四元组：{h['quadruples_evaluated']} 次；独立随机互异节点对：{h['pairs_evaluated']} 次。",
                          rf"- 平均四点偏差 $\delta_{{\mathrm{{avg}}}} = {h['delta_avg']:.6f}$。",
                          rf"- 平均最短路距离 $d_{{\mathrm{{avg}}}} = {h['d_avg']:.6f}$。",
                          rf"- 归一化平均双曲度 $\delta_G = \frac{{2\delta_{{\mathrm{{avg}}}}}}{{d_{{\mathrm{{avg}}}}}} = {h['delta_G']:.6f}$。",
                          rf"- $\delta_G$ 的近似蒙特卡洛标准误：{h['delta_G_approx_mc_se']}。",
                          "节点从整个分析分量直接均匀抽样；最短路在完整分量上精确计算，不使用候选节点池。",
                          r"该归一化公式不保证结果在 $[0,1]$，未做截断。四节点环可得到 $\delta_G\approx1.5$；完全图的四点偏差为 0，因此小值不能单独证明树状结构或层级性。"])
    lines.extend(["", "## 幂律特征", ""])
    for name,stats in report["degree_distributions"].items():
        fit=stats["power_law"]
        if not fit:
            lines.append(f"- {name}：非恒定尾部样本不足，未拟合。")
            continue
        p=fit["bootstrap"]["p_value"]
        p_text = str(p) if p is not None else r"\mathrm{N/A}"
        verdict="未完成拟合优度检验，不能判断是否服从幂律" if p is None else "在 0.1 阈值拒绝幂律尾部" if p<.1 else "未拒绝幂律尾部，但不等于证明幂律"
        lines.append(rf"- {name}：$\alpha={fit['alpha']:.3f}$，$x_{{\min}}={fit['xmin']}$，尾部 $n={fit['tail_nodes']}$，$\mathrm{{KS}}={fit['ks']:.4f}$，bootstrap $p={p_text}$；{verdict}。")
    lines.extend(["", "零度节点不参与幂律尾部拟合，其数量另行报告。阈值采用候选搜索中的最小 KS，候选数量见 JSON。",
                  "指数分布比较仅是一个替代模型，未排除对数正态等解释。网络节点度数相关，bootstrap 和似然比的 iid 假设是限制。",
                  f"本次 bootstrap 请求 {report['parameters']['bootstrap']} 次；默认 1000 次，可用 --bootstrap 0 跳过。接近 0.1 阈值时应增加重复次数并检查蒙特卡洛误差。",
                  "", "参考：", "", *[f"- {url}" for url in REFERENCES.values()], ""])
    return "\n".join(lines)


def publish_result(source, target):
    """保留逐文件替换，同时让新文件继承输出目录的权限。

    Windows 上 TemporaryDirectory 可能带有仅创建者可访问的 ACL。
    直接将其中的文件移动出去会保留受限 ACL，导致 VS Code 无法读取。
    因此在目标目录中用普通 open 创建新文件，仅复制内容，再替换目标文件。
    不使用 mkstemp/copy2，以免再次创建受限文件或复制源文件的元数据。
    """
    pending = target.parent / f".analyse-{uuid.uuid4().hex}.tmp"
    # 普通新文件继承目标目录权限；排他创建避免覆盖同名文件。
    created = False
    try:
        with pending.open("xb") as output:
            created = True
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
        pending.replace(target)
    finally:
        if created:
            pending.unlink(missing_ok=True)


def run(args):
    import numpy as np
    import networkx as nx
    graph,meta=load_graph(args)
    # 分析结果必须独立于输入目录，避免覆盖数据文件。
    if args.output_dir == Path(meta["path"]).parent or args.output_dir.is_relative_to(Path(meta["path"]).parent):
        raise ValueError("请将分析输出放在输入数据目录之外")
    print(f"Loaded {args.dataset}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",flush=True)
    tree,hierarchy=structure_metrics(graph,args.clustering_samples,args.seed)
    hyperbolicity = hyperbolicity_metrics(graph, args)
    undirected=graph.to_undirected()
    degree_sets={"undirected":list(dict(undirected.degree()).values())}
    if graph.is_directed():
        degree_sets.update({"in_degree":list(dict(graph.in_degree()).values()),
                            "out_degree":list(dict(graph.out_degree()).values())})
    distributions={}
    for i,(name,values) in enumerate(degree_sets.items()):
        print(f"Fitting {name} (bootstrap={args.bootstrap})...",flush=True)
        distributions[name]=degree_analysis(values,args,args.seed+i)
    report={"dataset":args.dataset,"input":meta,"seed":args.seed,
            "parameters":{"min_tail":args.min_tail,"max_xmin_candidates":args.max_xmin_candidates,
                          "bootstrap":args.bootstrap,"clustering_samples":args.clustering_samples},
            "versions":{"numpy":np.__version__,"networkx":nx.__version__},
            "tree_structure":tree,"hierarchy":hierarchy,"degree_distributions":distributions,
            "hyperbolicity": hyperbolicity,
            "references":REFERENCES}
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent,prefix=".analyse-") as temp:
        staging=Path(temp).resolve()
        if staging.parent != args.output_dir.parent:
            raise RuntimeError("Unexpected temporary directory")
        (staging/"analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        (staging/"summary.md").write_text(summary_text(report),encoding="utf-8")
        if not args.no_plots:
            write_plot(report,staging/"diagnostics.png")
            write_degree_plots(report, staging)
        if args.output_dir.exists() and not args.overwrite:
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(exist_ok=True)
        for name in ("analysis.json", "summary.md", *PLOT_FILES):
            target=args.output_dir/name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"输出文件冲突：{target}")
        for source in staging.iterdir():
            publish_result(source, args.output_dir/source.name)
        # 覆盖运行且关闭绘图时，移除同名旧图，防止将旧图误认作本次结果。
        if args.no_plots:
            for name in PLOT_FILES:
                (args.output_dir/name).unlink(missing_ok=True)
    print(f"Saved analysis to {args.output_dir}")
    return report


def main(argv=None):
    args=parse_args(argv)
    try:
        run(args)
    except ImportError as exc:
        print(f"缺少依赖：{exc}. 请安装 numpy scipy networkx matplotlib；PT 另需 torch torch-geometric。",file=sys.stderr)
        return 1
    except (OSError,ValueError,RuntimeError) as exc:
        print(f"分析失败：{exc}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
