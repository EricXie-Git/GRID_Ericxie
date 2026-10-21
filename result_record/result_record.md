# 20261019 
## Hyperbolic decoder test on twitter dataset
### (1) origin (do nothing)

```bash
python main.py --dataset twitter --decoder_type none
```

result:
-------Final Results-------
 recall@50: 0.25304237
 recall@100: 0.29406357
 ndcg@50: 0.19308366
 ndcg@100: 0.20712100


### (2) euclidean (use euclidean distance)

```bash
python main.py --dataset twitter --decoder_type euclidean --decoder_dim 32 --decoder_weight 0.1
```

result:
-------Final Results-------
 recall@50: 0.04568880
 recall@100: 0.06938837
 ndcg@50: 0.01751610
 ndcg@100: 0.02400175


### (3) euclidean (use euclidean distance but weight 0)

```bash
python main.py --dataset twitter --decoder_type euclidean --decoder_dim 32 --decoder_weight 0
```

result:
-------Final Results-------
 recall@50: 0.24967876
 recall@100: 0.29867974
 ndcg@50: 0.19173884
 ndcg@100: 0.20740767



### (4) hyperbolic (use hyperbolic distance)

```bash
python main.py --dataset twitter --decoder_type hyperbolic --decoder_dim 32 --decoder_weight 0.1
```

result:
-------Final Results-------
 recall@50: 0.04568880
 recall@100: 0.06925987
 ndcg@50: 0.01752011
 ndcg@100: 0.02395583

本轮实验在 Twitter 数据集上，为 GRID 增加了欧氏/双曲距离评分分支，得到以下结论：
| 实验设置 | Recall@50 | NDCG@50 |
|---|---:|---:|
| 原 GRID | 0.25304 | 0.19308 |
| 欧氏分支，权重 0 | 0.24968 | 0.19174 |
| 欧氏分支，权重 0.1 | 0.04569 | 0.01752 |
| 双曲分支，权重 0.1 | 0.04569 | 0.01752 |


结果分析：
1. 直接加入距离评分并从头联合训练，效果显著下降。 欧氏与双曲分支退化程度相近，说明问题更可能位于共同的融合与训练机制，而非双曲几何特有的问题。
2. 零权重实验基本恢复基线。 新增模块的初始化和随机数变化，不足以解释严重退化；问题与分支实际参与评分和梯度更新有关。
3. 同一联合训练模型在推理时关闭分支，效果没有恢复，反而进一步下降。 欧氏实验最佳检查点的 Recall@50 从 0.04569 降至 0.02406。这说明问题不只是辅助分数直接破坏排序，也涉及联合训练过程或分支间的共同适应。
4. 延长早停耐心没有明显改善。 根据新增实验反馈，将双曲实验的 patience 从 5 增至 10 后效果仍差，训练时长不足不再是优先解释。
5. 损失配合不合理是当前重点怀疑方向，但尚未证实。 原混合损失直接作用于融合分数，负距离分支可能通过整体降低分数来改善 BCE，却未改善正负用户排序。现有日志只记录总损失，无法确认具体原因。

后续应记录 BCE、BPR、CE 分项损失及正负例距离，并尝试从已训练的 GRID 出发，冻结主干、单独训练距离分支，以区分评分能力与联合优化问题。目前只能说明当前融合训练方案无效，不能据此否定双曲评分的可行性。



# 20261021
## 沿着上面思路做验证实验
### 冻结主干的诊断实验（同README中描述）

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

#### 1. 训练并保存基线结果：
    -------Final Results-------
    recall@50: 0.25636348
    recall@100: 0.29950184
    ndcg@50: 0.19440149
    ndcg@100: 0.20842667

#### 2. 冻结主干，使用原融合损失
    -------Final Results-------
    recall@50: 0.25937551
    recall@100: 0.30010951
    ndcg@50: 0.19537196
    ndcg@100: 0.20907128

#### 3. 冻结主干，使用原融合损失，将 fused 改为 bpr，仅优化评分分支的 BPR
    -------Final Results-------
    recall@50: 0.25967318
    recall@100: 0.30165228
    ndcg@50: 0.19601823
    ndcg@100: 0.20996214

#### 4. 冻结主干，使用原融合损失，仅优化评分分支的 BPR，但使用的是欧式
    -------Final Results-------
    recall@50: 0.25858086
    recall@100: 0.30193621
    ndcg@50: 0.19551910
    ndcg@100: 0.20967774


- `--decoder_diagnostics`：记录分项损失、正负例距离和评分尺度。`distance_gap` 为负例距离减正例距离；`objective` 为实际优化的损失。
- `--compare_decoder`：同一次前向计算分支开启/关闭的指标（ON/OFF）。
- `--freeze_backbone`：仅更新 Decoder，固定主干及评估随机性；OFF 指标应保持不变。
- `--decoder_objective bpr`：仅冻结模式可用。日志里的 BCE/BPR/CE 仍描述融合评分，但只有 Decoder 自身的 BPR 参与优化。
- `--save_checkpoint --ckpt_file 路径`：保存最佳模型；加载只恢复权重，不恢复优化器。输出路径应与基线不同。

欧氏对照将 `hyperbolic` 改为 `euclidean`。诊断联合训练时，只添加 `--decoder_diagnostics --compare_decoder`，不冻结主干。对比实验应保持数据划分、随机性及早停设置一致。
