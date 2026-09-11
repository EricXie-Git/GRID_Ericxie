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

Training records receive up to `--neg-num 100` distinct negatives sampled with
`--seed 21`, excluding the entire cleaned cascade and special IDs. Complete
histories, targets, and timestamps are saved. `preprocess_report.json` records
parameters, input/output hashes, cleaning and length statistics, graph statistics,
and compatibility notes. `sample_provenance.json` maps output records to their
source file and zero-based index.

**Training compatibility:** the current loader still truncates histories to
`--max_len` (default 200) and targets to 20. For full-target experiments, change
the loader to dynamically pad labels and choose a history limit large enough
(500 covers the supplied datasets). Preprocessing does not change the loader
or fix the existing test-set-based early stopping. Android and Christian also
need entries in `config.py`; the report provides the required `user_num`.

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
