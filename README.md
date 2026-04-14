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
