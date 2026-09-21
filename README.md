# GRID: Group-based Information Diffusion Prediction over Long-Context Cascades

[![WWW 2026](https://img.shields.io/badge/WWW-2026-blue)](https://doi.org/10.1145/3774904.3792082)


> **Many Hands Make Light Work: Group-based Information Diffusion Prediction over Long-Context Cascades**
> Accepted by *Proceedings of the ACM Web Conference 2026 (WWW '26)*


## Repository Layout

```text
GRID/
├── config.py
├── dataLoader.py
├── main.py
├── model.py
├── module.py
├── scripts/
│   ├── Efficiency_test.py
│   └── Group_test.py
├── utils/
│   ├── Metric.py
│   ├── Optim.py
│   └── Setup.py
└── dataset/
    └── twitter/
```

## Dependencies

The codebase depends primarily on:

- `torch`
- `torch-geometric`
- `numpy`
- `thop` for `scripts/Efficiency_test.py` only

Example installation:

```bash
pip install torch torch-geometric numpy thop
```

## Data Layout

The training pipeline expects the following files under `dataset/<dataset>/`:

```text
dataset/<dataset>/
├── cascade_train_neg.json
├── cascade_val.json
├── cascade_test.json
└── graph.pt
```

Each cascade file is a JSON list of records with the following structure:

```json
[
  {
    "observed": [1, 12, 42],
    "label": [56, 78],
    "neg": [9, 18, 24]
  }
]
```

- `observed`: observed user sequence used as model input
- `label`: target user set for the next diffusion step
- `neg`: candidate negative users used by the ranking loss

`graph.pt` should be a PyTorch Geometric `Data` object containing the social graph.

This repository currently includes the `twitter` dataset layout. `config.py` also contains dataset-specific presets for `douban`, `quora`, and `weibo`; those datasets should follow the same directory structure if added locally.

## Preprocessing Full Cascades

`data_preprocess.py` converts `cascade_train.json`, `cascade_valid.json`,
`cascade_test.json` (each record has `cascade` and `timestamp`) and a sparse
`graph.npz` into the four GRID input files. Install its dependencies in your
training environment:

```bash
pip install numpy scipy torch torch-geometric
python data_preprocess.py --dataset douban --dry-run
python data_preprocess.py --dataset douban
```

The default source is `dataset/origin_data/douban`, and the destination is
`dataset/douban`. Existing output directories require explicit `--overwrite`,
or use `--output-dir` to create a separate version:

```bash
python data_preprocess.py --dataset douban --overwrite
```

Overwrite mode replaces only the three cascade JSON files, `graph.pt`,
`preprocess_report.json`, and `sample_provenance.json`, preserving other files.
Results are generated and verified in a temporary directory before publication;
existing files are then replaced individually, not as a single transaction.
A dry run writes nothing, even when combined with `--overwrite`.

The default `--protocol paper` retains each user's first adoption, filters
cascades with fewer than 10 distinct participants, and uses the first 90% of
participants as history. These rules follow Appendix A of the
[paper](https://www.comp.hkbu.edu.hk/~xinhuang/publications/pdfs/WWW2026-Diffusion.pdf).
The configurable `--split-ratios 0.6 0.1 0.3` default is inferred from the
repository's Twitter files, not confirmed as a paper requirement. All source
splits are merged, cleaned, stably sorted by cascade start time, and repartitioned.
Boundaries are rounded down. To use 80% history and retain short cascades as
in the shipped Twitter data, select `--protocol repository` (minimum length 2).
Both presets remove repeated users within a cascade and retain duplicate full
records, which are reported.

The graph converter selects `(value & 1) != 0`, removes self-loops, preserves
edge direction, and writes unit edge attributes. This rule exactly recovers the
shipped Twitter edge set; confirm the relation encoding before applying it to
other source graphs. `--graph-relation-bit` and `--max-user-id` allow explicit
overrides. User IDs are preserved, 0 is PAD, and the trailing source special
node is excluded. Known user ranges come from `config.py`; for other datasets,
the fallback assumes a square graph of size `real_users + 2`.

For the supplied Weibo layout, use:

```bash
python data_preprocess.py --dataset weibo --dry-run
python data_preprocess.py --dataset weibo
```

This reads only `dataset/origin_data/Weibo/cascade_train.json` and `graph.txt`,
then writes `dataset/weibo`. The single JSON file is treated as a cascade pool:
it is cleaned, sorted by cascade start time, and split using `--split-ratios`.
The TXT graph must contain two integer user IDs per nonempty line. Its direction
is preserved, duplicate edges and self-loops are removed, and no relation-bit
filter is applied. Non-default `--graph-relation-bit` values are rejected for
Weibo. Its real user range is `1..31061` from `config.py` (overridable with
`--max-user-id`); `graph.pt` includes 31062 node slots including PAD.
Profiles, prompts, `gt.json`, bias attention, and co-occurrence edges are not
used. Dataset arguments `weibo`, `Weibo`, and `WEIBO` select the same preset.

Training records receive up to `--neg-num 100` distinct negatives sampled with
`--seed 21`, excluding the entire cleaned cascade and special IDs. Complete
histories, targets, and timestamps are saved. `preprocess_report.json` records
parameters, input/output hashes, cleaning and length statistics, graph statistics,
and compatibility notes. `sample_provenance.json` maps output records to their
source file and zero-based index.

**Training compatibility:** the current loader still truncates histories to
`--max_len` (default 200) and targets to 20. For full-target experiments, change
the loader to dynamically pad labels and choose a history limit large enough
(500 covers the supplied Twitter/Douban/Android/Christian data; the supplied
Weibo data needs up to 7369 observed users and 819 target users under the paper
preset). Preprocessing does not change the loader
or fix the existing test-set-based early stopping. Android and Christian also
need entries in `config.py`; the report provides the required `user_num`.

## Graph Structure Analysis

`data_analyse.py` analyses the **social graph**, not a cascade transmission tree.
User sequences alone do not specify who infected whom. Install the analysis
dependencies in your Python environment:

```bash
pip install numpy scipy networkx matplotlib
python data_analyse.py --dataset weibo
python data_analyse.py --dataset douban --no-plots
python data_analyse.py --dataset twitter
```

Twitter is supported by the same analysis pipeline: by default it reads
`dataset/origin_data/twitter/graph.npz`, selects relation bit 1, and uses real
user IDs 1 through 12627 from `config.py`. No preprocessing is required.
Results are saved to `analysis/twitter/`. To analyse the training graph instead,
use `python data_analyse.py --dataset twitter --graph-path dataset/twitter/graph.pt`.

Inputs are automatically located under `dataset/origin_data`, `dataset/origin`,
then `dataset`, preferring raw TXT/NPZ files to PT. Use `--graph-path` to select
a specific file. PT requires `torch` and `torch-geometric` and must come from a
trusted source. NPZ selects relation bit 1 by default; TXT/PT do not use bit
filtering. PAD/special nodes, self-loops and duplicate edges are removed.
Known user ranges come from `config.py`; specify `--max-user-id` for other
numbering conventions. All metrics use unweighted simple graphs.

The output directory defaults to `analysis/<dataset>` and contains:

- `summary.md`: Chinese explanation of results and limitations.
- `analysis.json`: all metrics, distributions, parameters and source hash.
- `diagnostics.png`: degree CCDF, clustering by degree, core numbers and BFS levels
  (unless `--no-plots` is used).
- `degree_distribution_linear.png`: empirical degree probabilities on linear axes,
  including zero-degree nodes.
- `degree_distribution_loglog.png`: the same probabilities on log-log axes;
  zero-degree nodes are omitted and their fraction is annotated. Both plots use
  all nodes as the denominator, without binning or fitting. Directed graphs have
  separate panels for undirected projection degree, in-degree and out-degree.
  `--no-plots` skips all three PNG files. Exact counts are stored in each
  distribution's `degree_histogram` in `analysis.json`.

Hierarchy diagnostics include k-core decomposition, sampled clustering versus
degree, and, for directed graphs, strongly connected components and condensation
DAG levels. Tree diagnostics use the undirected projection: leaves, bridges,
tree components and cycle rank `m - n + components`. BFS layers depend on the
selected root; condensation is always a DAG. Neither proves causal hierarchy.
`--direction auto` treats a completely reciprocal edge set as undirected; use
`--direction directed` or `undirected` to override that convention.

Degree tails are fitted with discrete maximum likelihood, choosing `xmin` by
minimum discrete KS distance over at most 64 candidate thresholds. The report
also compares the fitted tail to a discrete exponential distribution. This
comparison alone does not establish a power law. By default goodness-of-fit
bootstrap runs **1000 simulations** (each simulated sample is refitted).
Use `--bootstrap 0` to skip it for a quick structural analysis. For example:

```bash
python data_analyse.py --dataset weibo --bootstrap 1000 --overwrite
```

`--min-tail`, `--max-xmin-candidates` (0 searches all), `--clustering-samples`
(0 uses all nodes), and `--seed` control the analysis. A non-rejected power-law
tail is not proof of a power law, and network degrees violate strict iid
assumptions. See [Clauset et al.](https://arxiv.org/abs/0706.1062) and
[Ravasz and Barabasi](https://arxiv.org/abs/cond-mat/0206130) for the statistical
and clustering-scaling background. Existing reports require `--overwrite`;
other files in the output directory are preserved.

## Training and Testing

Run training with:

```bash
python main.py --dataset twitter
```

Common arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `twitter` | Dataset preset and data directory to use |
| `--batch_size` | `32` | Mini-batch size |
| `--max_len` | `200` | Maximum observed cascade length after truncation/padding |
| `--max_epochs` | `30` | Number of training epochs |
| `--group_num` | `20` | Number of grouped tokens produced by clustering |
| `--gnn_type` | `lightgcn` | Social graph encoder backbone |
| `--neg_num` | `100` | Maximum negatives sampled per cascade |
| `--metric_k` | `[50, 100]` | Evaluation cutoffs for Recall@K and NDCG@K |



## Additional Scripts

Check GroupSoftMax consistency:

```bash
python scripts/Group_test.py
```

Profile parameters, FLOPs, memory, and inference time:

```bash
python scripts/Efficiency_test.py --dataset twitter
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{GRID_WWW2026,
  author    = {Feng, Zihan and Yang, Yajun and Huang, Xin and Wang, Xin and Gao, Hong and Hu, Qinghua},
  title     = {Many Hands Make Light Work: Group-based Information Diffusion Prediction over Long-Context Cascades},
  year      = {2026},
  doi       = {10.1145/3774904.3792082},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  pages     = {4472--4481},
  numpages  = {10},
  location  = {United Arab Emirates},
  series    = {WWW '26}
}
```
