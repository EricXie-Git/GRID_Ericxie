r"""将完整传播级联 JSON 和稀疏关系图 NPZ 转换成 GRID 的输入文件。

使用示例（默认路径以本脚本所在的项目根目录为基准）：

    # 只检查数据并打印统计，不生成文件
    python data_preprocess.py --dataset douban --dry-run
    # 按默认协议处理 Douban，输出到新目录 dataset/douban
    python data_preprocess.py --dataset douban
    # 输出目录已存在时，显式允许替换本脚本生成的文件，保留其他文件
    python data_preprocess.py --dataset douban --overwrite
    # 使用仓库兼容协议，将 Twitter 输出到另一个目录
    python data_preprocess.py --dataset twitter --protocol repository \
        --output-dir dataset/twitter_reprocessed

依赖：numpy、scipy、torch、torch-geometric。

处理流程：读取并合并原数据 → 清洗级联 → 按时间划分数据集 → 切分观测/标签
          → 为训练集采负样本 → 提取图关系 → 保存文件与辅助报告。

两套协议的区别：
  paper：按论文附录保留每个用户的首次参与，过滤不足 10 人的级联，前 90% 为观测。
  repository：参考仓库现有 Twitter 文件，保留至少 2 人的级联，前 80% 为观测。
两者都默认按级联开始时间划分为 60% 训练、10% 验证、30% 测试，可通过参数修改。
这个数据集划分比例来自仓库文件的实际统计，不能视为已确认的论文要求。
所有比例边界向下取整。单条级联内部的重复用户会删除；整条重复记录只统计、不删除。

图转换默认提取最低位关系（value & 1），删除自环，保留原始边方向。
该规则能还原仓库 Twitter 的边集合；其他数据集的关系含义需结合来源建图代码确认。
用户 ID 不重新编号，0 留给 PAD，原图末尾的特殊节点不进入真实用户范围。
此脚本不截断观测或标签，也不合成超长级联；模型加载器仍有其自身的长度限制。

主要函数：load_cascades 清洗；make_splits 划分及负采样；convert_graph 转图；run 串联流程。
report、校验值和溯源文件仅用于检查/复现，训练只读取三个级联 JSON 和 graph.pt。
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile


# 默认输入、输出路径以项目根目录为基准，避免受启动命令所在目录影响。
ROOT = Path(__file__).resolve().parent
# 两个元组的位置一一对应训练、验证、测试；GRID 的验证文件名使用 val 而非 valid。
INPUT_FILES = ("cascade_train.json", "cascade_valid.json", "cascade_test.json")
OUTPUT_FILES = ("cascade_train_neg.json", "cascade_val.json", "cascade_test.json")
# 覆盖模式只替换这六个文件，不清空输出目录或修改其他文件。
GENERATED_FILES = (*OUTPUT_FILES, "graph.pt", "preprocess_report.json", "sample_provenance.json")
# 仅作为报告中的参考来源，不会联网下载论文，也不参与数据处理。
PAPER_URL = "https://www.comp.hkbu.edu.hk/~xinhuang/publications/pdfs/WWW2026-Diffusion.pdf"


def fraction(value):
    """将命令行中的 0.9 或 9/10 转为精确分数，避免浮点误差影响切分位置。"""
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("Expected a fraction or decimal, e.g. 0.9") from exc


def parse_args(argv=None):
    """读取参数、补齐协议默认值，并检查比例、用户范围及目录是否合法。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # 数据来源和输出位置；显式传入的相对路径仍相对于当前工作目录。
    parser.add_argument("--dataset", default="douban", help="Dataset name (default: douban)")
    parser.add_argument("--input-dir", type=Path, help="Default: dataset/origin_data/<dataset>")
    parser.add_argument("--output-dir", type=Path, help="Output directory; default: dataset/<dataset>")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated files in an existing output directory; keep other files")
    # observed-ratio 决定一条级联如何切分；split-ratios 决定整批样本如何划分。
    # 两者是不同层面的比例，不要混淆。
    parser.add_argument("--protocol", choices=("paper", "repository"), default="paper")
    parser.add_argument("--observed-ratio", type=fraction,
                        help="Override paper=0.9 / repository=0.8")
    parser.add_argument("--min-cascade-len", type=int,
                        help="Minimum distinct users; paper=10 / repository=2")
    parser.add_argument("--split-ratios", type=fraction, nargs=3,
                        default=tuple(map(Fraction, ("0.6", "0.1", "0.3"))),
                        metavar=("TRAIN", "VALID", "TEST"), help="Positive ratios summing to 1")
    # 训练负样本数量和随机种子；最大用户 ID 不包含 PAD 和末尾特殊节点。
    parser.add_argument("--neg-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--max-user-id", type=int,
                        help="Default: config.py preset, otherwise source graph size minus 2")
    parser.add_argument("--graph-relation-bit", type=int, default=1,
                        help="Single relation bit in the integer NPZ values (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing")
    args = parser.parse_args(argv)
    if not args.dataset or Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
        parser.error("--dataset must be a simple directory name")
    # 手动指定的参数优先；未指定时才采用 paper/repository 的默认值。
    args.observed_ratio = (args.observed_ratio if args.observed_ratio is not None
                           else Fraction(9 if args.protocol == "paper" else 8, 10))
    args.min_cascade_len = (args.min_cascade_len if args.min_cascade_len is not None
                            else 10 if args.protocol == "paper" else 2)
    if not 0 < args.observed_ratio < 1:
        parser.error("--observed-ratio must be between 0 and 1")
    if args.min_cascade_len < 2 or args.neg_num < 1:
        parser.error("Minimum cascade length must be >=2 and --neg-num must be positive")
    if any(r <= 0 for r in args.split_ratios) or sum(args.split_ratios) != 1:
        parser.error("--split-ratios must be positive and sum exactly to 1")
    # 关系位必须是单个二进制位，如 1、2、4；bit & (bit - 1) 用来判断是否为 2 的幂。
    bit = args.graph_relation_bit
    if bit < 1 or bit & (bit - 1):
        parser.error("--graph-relation-bit must be a power of two (1, 2, 4, ...)")
    if args.max_user_id is not None and args.max_user_id < 1:
        parser.error("--max-user-id must be positive")
    args.input_dir = (args.input_dir or ROOT / "dataset/origin_data" / args.dataset).resolve()
    args.output_dir = (args.output_dir or ROOT / "dataset" / args.dataset).resolve()
    # 防止输出覆盖原始数据；只读预检允许检查已经生成过的数据集。
    if (args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir)
            or args.input_dir.is_relative_to(args.output_dir)):
        parser.error("Input and output directories must not overlap")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("Output path exists but is not a directory")
    if args.output_dir.exists() and not (args.dry_run or args.overwrite):
        parser.error("Output already exists; use --overwrite or choose a new --output-dir.")
    return args


def publish_outputs(payload, output_dir, overwrite):
    """发布已验证的结果；覆盖模式仅替换固定输出文件，保留目录中的其他内容。"""
    # 先确认新文件完整，避免在写入中途才发现缺少文件。
    for name in GENERATED_FILES:
        if not (payload / name).is_file():
            raise FileNotFoundError(f"Missing generated file: {name}")
    if not output_dir.exists():
        payload.rename(output_dir)
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output is not a directory: {output_dir}")
    if not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite: {output_dir}")
    # 在替换前检查全部目标，拒绝同名目录或符号链接，不递归删除任何内容。
    for name in GENERATED_FILES:
        target = output_dir / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Cannot overwrite a directory or symbolic link: {target}")
    # 同一磁盘内逐文件替换，避免先删除旧文件再写新文件。
    # 每个文件独立替换；这不是六个文件整体的原子事务。
    for name in GENERATED_FILES:
        (payload / name).replace(output_dir / name)


def checksum(path):
    """分块计算文件的 SHA-256 校验值，用来确认文件是否改变；与模型训练无关。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cascades(input_dir, max_user_id, min_length):
    """合并原始三个集合，按时间清洗每条级联，再按级联开始时间排序。

    返回 rows（清洗后的用户、时间及原始位置）和 stats（清洗数量统计）。
    此处不沿用原来的训练/验证/测试边界，后续由 make_splits 重新划分。
    """
    rows = []
    stats = Counter()
    # 以完整的“用户-时间”序列统计重复记录，仅报告，不在这里删除整条样本。
    fingerprints = Counter()
    for name in INPUT_FILES:
        with (input_dir / name).open(encoding="utf-8") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            raise ValueError(f"{name}: top-level JSON must be a list")
        stats[name] = len(records)
        for index, record in enumerate(records):
            context = f"{name}[{index}]"
            if not isinstance(record, dict):
                raise ValueError(f"{context}: expected an object")
            # 用户与时间必须逐项对应；拒绝越界 ID、PAD 和无效时间戳。
            users, times = record.get("cascade"), record.get("timestamp")
            if not isinstance(users, list) or not isinstance(times, list) or len(users) != len(times):
                raise ValueError(f"{context}: cascade/timestamp must be equally sized lists")
            if any(type(u) is not int or not 1 <= u <= max_user_id for u in users):
                raise ValueError(f"{context}: user IDs must be integers in 1..{max_user_id}")
            if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
                raise ValueError(f"{context}: timestamps must be finite numbers")
            # 把用户和时间绑在一起排序，避免对应关系错位。
            # Python 的排序是稳定的：相同时间戳的参与者保持原相对顺序。
            pairs = list(zip(users, times))
            ordered = sorted(pairs, key=lambda pair: pair[1])
            stats["reordered_cascades"] += ordered != pairs
            # 排序后第一次遇到该用户就是其首次参与，同时保留对应的时间戳。
            seen = set()
            unique = []
            for user, timestamp in ordered:
                if user not in seen:
                    unique.append((user, timestamp))
                    seen.add(user)
            stats["removed_repeat_adoptions"] += len(ordered) - len(unique)
            stats["cascades_with_repeat_adoptions"] += len(ordered) != len(unique)
            # 长度阈值基于去重后的真实参与人数，而不是原始记录条数。
            if len(unique) < min_length:
                stats["filtered_short_cascades"] += 1
                continue
            fingerprints[tuple(unique)] += 1
            # source_index 从 0 开始，便于从处理结果追溯到原始 JSON 中的记录。
            rows.append({"users": [p[0] for p in unique], "times": [p[1] for p in unique],
                         "source_file": name, "source_index": index})
    # 按每条级联的起始时间排序，随后把较早的样本用于训练、较晚的用于测试。
    rows.sort(key=lambda row: row["times"][0])
    stats["retained_cascades"] = len(rows)
    stats["duplicate_records_retained"] = sum(n - 1 for n in fingerprints.values())
    return rows, dict(stats)


def make_splits(rows, ratios, observed_ratio, max_user_id, neg_num, seed):
    """生成三个 GRID 样本集合，并保存每条样本的来源位置。

    rows 已按开始时间排好序。先划分集合，再将每条级联切成 observed/label。
    仅训练集生成 neg；观测、标签和时间戳均保留完整长度，不做补齐或截断。
    """
    n = len(rows)
    # 例如默认比例下，first=floor(N*0.6)，second=floor(N*0.7)。
    # 最后一段取全部剩余样本，避免取整导致漏样本。
    first, second = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    partitions = (rows[:first], rows[first:second], rows[second:])
    if any(not part for part in partitions):
        raise ValueError("Cleaning/split ratios produce an empty train, validation or test set")
    # 独立随机数生成器：同样输入和种子得到相同负样本，不改变全局随机状态。
    rng = random.Random(seed)
    outputs, provenance = {}, {}
    for filename, partition in zip(OUTPUT_FILES, partitions):
        records, sources = [], []
        for row in partition:
            # 例如 15 人、观测比例 0.9：前 13 人为输入，剩余 2 人为预测标签。
            cut = int(len(row["users"]) * observed_ratio)
            if not 0 < cut < len(row["users"]):
                raise ValueError("Observed ratio produces an empty history or label; adjust the ratio/minimum length")
            item = {"observed": row["users"][:cut], "label": row["users"][cut:],
                    "observed_timestamp": row["times"][:cut], "label_timestamp": row["times"][cut:]}
            # 负样本从全部真实用户中抽取，排除整条级联的参与者（包括未来标签）。
            # ID 从 1 开始，因此不会抽到 PAD=0；range 上界也排除了末尾特殊节点。
            if filename == OUTPUT_FILES[0]:
                excluded = set(row["users"])
                candidates = [u for u in range(1, max_user_id + 1) if u not in excluded]
                if not candidates:
                    raise ValueError("A training cascade contains every user: no negative candidates")
                # sample 为无放回采样；候选不足指定数量时使用所有候选，不重复凑数。
                item["neg"] = rng.sample(candidates, min(neg_num, len(candidates)))
            records.append(item)
            sources.append({"file": row["source_file"], "index": row["source_index"]})
        outputs[filename], provenance[filename] = records, sources
    return outputs, provenance


def convert_graph(adj, max_user_id, relation_bit):
    """将稀疏关系矩阵转为 PyG Data，返回图对象和统计信息。

    输入边值是关系编码，输出 edge_attr 统一为 1；不将编码本身当作边权。
    保留原始行→列方向，不自动补反向边，也不转成占用大量内存的稠密矩阵。
    """
    import numpy as np
    import torch
    from torch_geometric.data import Data

    # COO 用 row、col、data 三个数组分别表示每条边的起点、终点、关系编码。
    coo = adj.tocoo(copy=True)
    if (not np.issubdtype(coo.data.dtype, np.number) or np.iscomplexobj(coo.data)
            or not np.isfinite(coo.data).all()
            or not np.equal(coo.data, np.floor(coo.data)).all()
            or (coo.data < 0).any() or (coo.data >= 2**63).any()):
        raise ValueError("Graph values must be nonnegative integer relation codes")
    values = coo.data.astype(np.int64)
    # 按位筛选关系：当 relation_bit=1 时，值为 1、3、5、7 的边均会保留。
    # 不能只判断 values == 1，否则会丢掉同时带有其他关系位的边。再去除自环。
    selected = ((values & relation_bit) != 0) & (coo.row != coo.col)
    valid = ((coo.row >= 1) & (coo.row <= max_user_id)
             & (coo.col >= 1) & (coo.col <= max_user_id))
    # 选中关系若连接了真实用户范围外的节点，说明编号假设可能有误，直接报错。
    if np.any(selected & ~valid):
        raise ValueError("Selected graph relation has non-self edges outside the real user range")
    # 先得到 [边数, 2] 并去重，再转置为 PyG 要求的 [2, 边数]。
    edges = np.unique(np.stack((coo.row[selected], coo.col[selected]), axis=1), axis=0)
    if not len(edges):
        raise ValueError("No graph edges remain; check --graph-relation-bit")
    edge_index = torch.from_numpy(edges.T.copy()).long()
    # num_nodes 包含编号 0 的 PAD 槽位；即使某个真实用户没有边，也保留其编号空间。
    graph = Data(edge_index=edge_index, edge_attr=torch.ones(len(edges)), num_nodes=max_user_id + 1)
    stats = {"source_shape": list(adj.shape), "source_nonzero_entries": int(adj.count_nonzero()),
             "relation_value_counts": {str(k): int(v) for k, v in Counter(values.tolist()).items()},
             "relation_bit": relation_bit, "num_nodes_including_pad": graph.num_nodes,
             "edge_count": graph.num_edges, "is_undirected": graph.is_undirected(),
             "self_loops_removed": int(np.count_nonzero(coo.row == coo.col))}
    return graph, stats


def length_stats(values):
    """汇总一组序列长度，供报告展示，不改变原始序列。"""
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def run(args):
    """主处理流程：检查输入 → 确定用户范围 → 转换数据 → 汇总报告 → 保存。"""
    # 将训练依赖放在函数内导入，使 --help 和参数解析无需安装 PyTorch 也能执行。
    try:
        import scipy.sparse as sparse
        import torch
        import torch_geometric
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Install into your training environment with: "
                           "python -m pip install numpy scipy torch torch-geometric") from exc
    from config import DATASET_CONFIGS

    # 第一步：确认三个级联文件和图文件齐全，并读取稀疏图。
    sources = [args.input_dir / f for f in (*INPUT_FILES, "graph.npz")]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    adj = sparse.load_npz(sources[-1])
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Source graph must be square")
    # 第二步：确定真实用户编号上限。优先级：命令行 > config.py > 图大小减 2。
    # “减 2”假定来源图额外包含 PAD 和末尾特殊节点；未知数据集应核对或手动指定。
    config = DATASET_CONFIGS.get(args.dataset)
    max_id = args.max_user_id
    id_source = "--max-user-id"
    if max_id is None:
        max_id = config.user_num - 1 if config else adj.shape[0] - 2
        id_source = "config.py" if config else "graph size minus PAD and final special node (assumption)"
    if not 1 <= max_id < adj.shape[0]:
        raise ValueError("max_user_id must fit within the source graph")
    # 第三步：执行实际的数据转换；下方报告部分不再改变这些样本或图。
    rows, cleaning = load_cascades(args.input_dir, max_id, args.min_cascade_len)
    outputs, provenance = make_splits(rows, args.split_ratios, args.observed_ratio,
                                     max_id, args.neg_num, args.seed)
    graph, graph_stats = convert_graph(adj, max_id, args.graph_relation_bit)
    # 第四步：统计输入/标签长度，提示现有加载器的 200/20 上限会影响多少条样本。
    split_stats = {}
    for name, records in outputs.items():
        split_stats[name] = {"count": len(records),
                             "observed_length": length_stats([len(r["observed"]) for r in records]),
                             "label_length": length_stats([len(r["label"]) for r in records]),
                             "observed_over_200": sum(len(r["observed"]) > 200 for r in records),
                             "labels_over_20": sum(len(r["label"]) > 20 for r in records)}
    warnings = ["60/10/30 is a repository-derived default, not a verified paper dataset-split requirement.",
                "Relation bit 1 matches shipped Twitter; verify relation semantics for other source datasets."]
    if args.graph_relation_bit != 1:
        warnings.append("Custom graph relation selected; this differs from the shipped Twitter graph rule.")
    if cleaning["duplicate_records_retained"]:
        warnings.append("Identical complete cascades were retained; inspect provenance before comparing splits.")
    if any(s["labels_over_20"] for s in split_stats.values()):
        warnings.append("Current dataLoader.py truncates labels to 20: use dynamic label padding for full-target evaluation.")
    if any(s["observed_over_200"] for s in split_stats.values()):
        warnings.append("Current --max_len=200 truncates histories; increase it to the reported maximum or use dynamic padding.")
    if not config or config.user_num != max_id + 1:
        warnings.append(f"Add/update config.py DATASET_CONFIGS[{args.dataset!r}] with user_num={max_id + 1}.")
    # report 是辅助记录：参数、文件校验值、软件版本、清洗/划分/图统计。
    # PAPER_URL 仅注明参考来源；这些字段不会被 GRID 训练代码读取。
    report = {"dataset": args.dataset, "protocol_preset": args.protocol,
              "input_dir": str(args.input_dir), "output_dir": str(args.output_dir),
              "paper_reference": PAPER_URL, "observed_ratio": str(args.observed_ratio),
              "min_cascade_len": args.min_cascade_len,
              "split_ratios": [str(r) for r in args.split_ratios],
              "boundary_rule": "floor N*train and N*(train+valid); floor L*observed_ratio",
              "max_user_id": max_id, "user_num": max_id + 1, "id_range_source": id_source,
              "neg_num": args.neg_num, "seed": args.seed,
              "negative_sampling": "uniform without replacement from 1..max_user_id excluding full cleaned cascade",
              "source_sha256": {p.name: checksum(p) for p in sources},
              "versions": {"python": sys.version, "torch": torch.__version__,
                           "torch_geometric": torch_geometric.__version__},
              "cleaning": cleaning, "splits": split_stats, "graph": graph_stats,
              "warnings": warnings}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # 只读预检仍执行清洗、采样和图转换，但结果仅在内存中，不向磁盘写入。
    if args.dry_run:
        return report

    # 第五步：先在输出目录旁边的临时目录完成写入和验证，再发布结果。
    # 新目录整体移动；已有目录仅在 --overwrite 时逐个替换本脚本生成的文件。
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grid-preprocess-", dir=args.output_dir.parent) as temp:
        staging = Path(temp).resolve()
        if staging.parent != args.output_dir.parent or not staging.name.startswith(".grid-preprocess-"):
            raise RuntimeError("Unexpected staging directory")
        payload = staging / "payload"
        payload.mkdir()
        for name, records in outputs.items():
            (payload / name).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        torch.save(graph, payload / "graph.pt")
        # 这里只重新读取本脚本刚保存的图，检查序列化前后的节点数、边和边属性一致。
        restored = torch.load(payload / "graph.pt", map_location="cpu", weights_only=False)
        if (restored.num_nodes != graph.num_nodes or not torch.equal(restored.edge_index, graph.edge_index)
                or not torch.equal(restored.edge_attr, graph.edge_attr)):
            raise RuntimeError("Graph serialization verification failed")
        # 训练需要的文件到此已生成。下面两个 JSON 分别用于复现实验和追溯原始样本。
        report["output_sha256"] = {name: checksum(payload / name) for name in (*OUTPUT_FILES, "graph.pt")}
        (payload / "preprocess_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (payload / "sample_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        publish_outputs(payload, args.output_dir, args.overwrite)
    print(f"Saved GRID dataset to {args.output_dir}")
    return report


def main(argv=None):
    """命令行入口：解析参数，调用处理流程；成功返回 0，处理失败返回 1。"""
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
