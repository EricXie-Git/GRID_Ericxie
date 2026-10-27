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

## 训练与评分分支

```bash
python main.py --dataset twitter
# 增加双曲评分分支；欧氏对照改为 euclidean
python main.py --dataset twitter --decoder_type hyperbolic --decoder_weight 0.1
```

默认关闭分支。启用后，评分为 $s=s_{\mathrm{GRID}}-\lambda d^2/\tau$，将级联和用户表示投影后计算欧氏或双曲距离，并使用融合评分训练。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--decoder_type` | `none` | `none` / `euclidean` / `hyperbolic` |
| `--decoder_dim` | `32` | 分支投影维度 |
| `--decoder_weight` | `0.1` | 融合权重，0 不参与融合 |
| `--decoder_temperature` | `1.0` | 距离评分温度 |
| `--selection_split` | `test` | 模型选择集；新实验建议设为 `val` |
| `--patience` | `5` | 早停耐心轮数 |

其他参数见 `python main.py --help`。为兼容旧流程，默认仍按测试集选择模型；指定 `--selection_split val` 后，用验证集选择模型，最后评估测试集。

### 冻结主干的诊断实验

先保存基线，再只训练评分分支，比较两种损失。以下命令在 Linux 项目根目录执行：

```bash
# 1. 训练并保存基线
python main.py --dataset twitter --selection_split val \
  --deterministic_eval --early_stop_after_warmup \
  --save_checkpoint --ckpt_file checkpoint/twitter_base.pt

# 2. 冻结主干，使用原融合损失
python main.py --dataset twitter --decoder_type hyperbolic \
  --load_checkpoint checkpoint/twitter_base.pt --freeze_backbone \
  --decoder_objective fused --selection_split val --early_stop_after_warmup \
  --decoder_diagnostics --compare_decoder

# 3. 重复第 2 条命令，将 fused 改为 bpr，仅优化评分分支的 BPR
```

- `--decoder_diagnostics`：记录分项损失、正负例距离和评分尺度。`distance_gap` 为负例距离减正例距离；`objective` 为实际优化的损失。
- `--compare_decoder`：同一次前向计算分支开启/关闭的指标（ON/OFF）。
- `--freeze_backbone`：仅更新 Decoder，固定主干及评估随机性；OFF 指标应保持不变。
- `--decoder_objective bpr`：仅冻结模式可用。日志里的 BCE/BPR/CE 仍描述融合评分，但只有 Decoder 自身的 BPR 参与优化。
- `--save_checkpoint --ckpt_file 路径`：保存最佳模型；加载只恢复权重，不恢复优化器。输出路径应与基线不同。

欧氏对照将 `hyperbolic` 改为 `euclidean`。诊断联合训练时，只添加 `--decoder_diagnostics --compare_decoder`，不冻结主干。对比实验应保持数据划分、随机性及早停设置一致。

### 独立损失训练（从头训练）

`separate` 使用 $L=L_{\mathrm{GRID}}(s_{\mathrm{GRID}})+\alpha L_{\mathrm{BPR}}(s_{\mathrm{distance}})$，预测时仍融合两套评分。

```bash
mkdir -p result_record
python -u main.py --dataset twitter --decoder_type hyperbolic \
  --decoder_objective separate --decoder_loss_weight 0.1 --decoder_weight 0.1 \
  --decoder_detach_backbone --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_detach.log
```

- `--decoder_loss_weight` 控制分支 BPR 权重；`--decoder_weight` 只控制评分融合。
- `--decoder_detach_backbone` 阻断分支梯度回传主干，主干仍由原损失训练；去掉它可做联合梯度对照。
- 本模式不与 `--freeze_backbone` 同用；欧氏对照改为 `euclidean`，并更换日志文件名。
- 日志中 `base_*` 对应原 GRID 评分，`decoder_bpr` 为分支损失，`objective` 为加权总损失。旧模式及默认行为保持不变。

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
