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
实验设置	Recall@50	NDCG@50
原 GRID	0.25304	0.19308
欧氏分支，权重 0	0.24968	0.19174
欧氏分支，权重 0.1	0.04569	0.01752
双曲分支，权重 0.1	0.04569	0.01752


1. 直接加入距离评分并从头联合训练，效果显著下降。 欧氏与双曲分支退化程度相近，说明问题更可能位于共同的融合与训练机制，而非双曲几何特有的问题。
2. 零权重实验基本恢复基线。 新增模块的初始化和随机数变化，不足以解释严重退化；问题与分支实际参与评分和梯度更新有关。
3. 同一联合训练模型在推理时关闭分支，效果没有恢复，反而进一步下降。 欧氏实验最佳检查点的 Recall@50 从 0.04569 降至 0.02406。这说明问题不只是辅助分数直接破坏排序，也涉及联合训练过程或分支间的共同适应。
4. 延长早停耐心没有明显改善。 根据新增实验反馈，将双曲实验的 patience 从 5 增至 10 后效果仍差，训练时长不足不再是优先解释。
5. 损失配合不合理是当前重点怀疑方向，但尚未证实。 原混合损失直接作用于融合分数，负距离分支可能通过整体降低分数来改善 BCE，却未改善正负用户排序。现有日志只记录总损失，无法确认具体原因。
后续应记录 BCE、BPR、CE 分项损失及正负例距离，并尝试从已训练的 GRID 出发，冻结主干、单独训练距离分支，以区分评分能力与联合优化问题。目前只能说明当前融合训练方案无效，不能据此否定双曲评分的可行性。结果分析：

