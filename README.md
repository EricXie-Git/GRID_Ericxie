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

## 数据预处理

将原始级联和社交图转换成 GRID 输入，支持 Douban、Weibo 等数据集：

```bash
pip install scipy
python data_preprocess.py --dataset douban --dry-run  # 检查输入
python data_preprocess.py --dataset douban
python data_preprocess.py --dataset weibo
```

输出到 `dataset/<dataset>/`，覆盖需加 `--overwrite`。自定义路径使用 `--input-dir`、`--output-dir`，完整选项见 `--help`。

默认 `--protocol paper`：用户去重，保留至少 10 人的级联，前 90% 为历史；按起始时间排序，以 6:1:3 划分训练、验证、测试集。划分比例来自仓库数据，非论文明确要求。`--protocol repository` 改为至少 2 人、前 80% 为历史。

- Douban：读取三个级联 JSON 和 `graph.npz`，默认保留关系位 1。
- Weibo：读取 `cascade_train.json` 和 `graph.txt`，其他辅助文件不参与处理。
- 负例排除整个级联的用户；用户 ID 保持不变，0 为 PAD。
- 输出另含处理报告和样本来源。训练加载器仍默认截断历史到 200 人、目标到 20 人。

## 图结构分析

分析社交图的层级性、树状特征、度分布及平均 δ-双曲性：

```bash
pip install scipy networkx matplotlib
python data_analyse.py --dataset douban
python data_analyse.py --dataset weibo
python data_analyse.py --dataset twitter
```

自动从 `dataset/origin_data`、`dataset/origin`、`dataset` 查找输入。指定训练图可加 `--graph-path dataset/twitter/graph.pt`。

结果位于 `analysis/<dataset>/`：

| 文件 | 内容 |
|---|---|
| `summary.md` / `analysis.json` | 结果解释 / 完整数值 |
| `diagnostics.png` | 度 CCDF、度与聚类系数、核数、BFS 层级 |
| `degree_distribution_linear.png` | 普通坐标度分布 |
| `degree_distribution_loglog.png` | 双对数坐标度分布 |

常用选项：`--overwrite` 覆盖结果，`--no-plots` 跳过绘图，`--bootstrap 0` 跳过默认 1000 次幂律检验，`--hyperbolicity off` 跳过双曲性计算。

幂律检验针对拟合选出的**尾部**，不能据此认定整个分布符合幂律。层级和树状指标描述社交图，不代表真实传播树。

双曲性默认在无向投影的最大连通分量上，随机采样 10 万个四元组和 10 万对节点，计算精确最短路：
$\delta=(S_{\max}-S_{\mathrm{mid}})/2$，$\delta_G=2\delta_{\mathrm{avg}}/d_{\mathrm{avg}}$。
可用 `--hyperbolicity-samples`、`--hyperbolicity-pairs` 调整采样数。这是平均统计量，非最大 Gromov 常数；归一化值不保证在 $[0,1]$，值小也不能单独证明树状结构。

方法参照：[1]谢文锦.社交与学术网络上的信息传播预测研究[D].西南大学,2025.DOI:10.27684/d.cnki.gxndx.2025.003879.

## 训练与测试

```bash
python main.py --dataset twitter
# 使用验证集选择模型，并保存最佳权重
python main.py --dataset twitter --selection_split val \
  --save_checkpoint --ckpt_file checkpoint/grid_best.pt
```

保留 GRID 原有图编码、分组、注意力、预测头和 BCE＋BPR＋CE 混合损失。额外距离评分分支及其专用开关已移除。

常用选项：`--patience` 设置早停轮数，`--early_stop_after_warmup` 从预热结束后开始累计早停，`--deterministic_eval` 关闭评估时的分组噪声。默认仍按原流程使用测试集选择模型，新实验建议指定 `--selection_split val`。

`--load_checkpoint 路径` 加载模型权重（不恢复优化器）。原 GRID 权重及此前保存的无分支基线可加载；含额外分支参数的历史检查点需配合备份代码复现实验。已有检查点、日志和结果记录均保留，保存新结果时使用新的文件名。

完整参数见 `python main.py --help`。

### 可选双曲图编码（第一步）

```bash
mkdir -p result_record
python -u main.py --dataset twitter --gnn_type lorentz_lightgcn \
  --selection_split val --deterministic_eval --early_stop_after_warmup \
  2>&1 | tee result_record/twitter_lorentz_lightgcn.log
```

默认仍为 `lightgcn`；对照实验只替换 `--gnn_type` 和日志文件名。
新增编码器借鉴 [Hgformer（ICML 2025）的 LHGCN](https://proceedings.mlr.press/v267/yang25o.html)：指数映射 → Lorentz 重心聚合 → 对数映射 → 原有层间加权融合。输出维度、欧氏分组、注意力、预测头和原损失保持不变，梯度可回传到用户嵌入。

本实现固定曲率为 −1，沿用 GRID 的边方向，不添加自环，无邻居节点回退到原点；dropout 在切空间执行后映射回来。因此它是适配版本，不是完整复现 Hgformer。几何内部用 float64，切向量范数上限为 5，输出恢复原 dtype；这会增加编码阶段的计算开销。重心采用 Lorentz 归一化，不是精确测地 Fréchet 均值。

需要保存最佳权重时追加 `--save_checkpoint --ckpt_file checkpoint/twitter_lorentz.pt`，默认仅保存上述终端日志。

## 辅助检查

```bash
python -m unittest discover -s tests -v       # 小图兼容性与训练流程测试
python scripts/Group_test.py                 # GroupSoftMax 检查
python scripts/Efficiency_test.py --dataset twitter  # 效率测试
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
