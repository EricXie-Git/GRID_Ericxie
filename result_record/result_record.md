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
2. 零权重实验基本恢复基线水平。 新增模块的初始化和随机数变化，不足以解释严重退化；问题与分支实际参与评分和梯度更新有关。
3. 同一联合训练模型在推理时关闭分支，效果没有恢复，反而进一步下降。 欧氏实验最佳检查点的 Recall@50 从 0.04569 降至 0.02406。这说明问题不只是辅助分数直接破坏排序，也涉及联合训练过程或分支间的共同适应。
4. 延长早停耐心没有明显改善。 根据新增实验反馈，将双曲实验的 patience 从 5 增至 10 后效果仍差，因此训练时长不足和性能下降没有太大关系。
5. 损失配合不合理是重点怀疑的方向，但尚未证实。 原混合损失直接作用于融合分数，负距离分支可能通过整体降低分数来改善 BCE，却未改善正负用户排序。现有日志只记录总损失，无法确认具体原因。

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


双曲 BPR 略优于欧氏 BPR，但目前没有表现出明显的优势。
| 设置 | Recall@50 | Recall@100 | NDCG@50 | NDCG@100 |
|---|---:|---:|---:|---:|
| GRID 基线 | 0.25636 | 0.29950 | 0.19440 | 0.20843 |
| 欧氏 BPR | 0.25858 | 0.30194 | 0.19552 | 0.20968 |
| 双曲 BPR | 0.25967 | 0.30165 | 0.19602 | 0.20996 |

主要结论：
1. 两种距离分支都能提供小幅收益。 因此，不能把双曲相对基线的全部提升归因于双曲几何，额外的投影和距离评分本身也有作用。
2. 双曲在前三项中的排序表现略好，但差距很小。 相比欧氏，Recall@50 高约 0.42%、NDCG@50 高约 0.26%（相对提升），Recall@100 则略低。更准确地说，是有一点靠前排名的优势，尚未体现明显优势。
3. 冻结主干解决严重退化，比更换空间带来的影响大得多。 结合此前结果，当前最有力的证据仍是：从头联合训练存在问题，而冻结后两种空间都能正常发挥辅助作用。

对现在的研究目标，初步结论：双曲评分方案可用，但仅把欧氏表示投影到双曲空间再计算距离，还不足以充分体现双曲结构的价值。 下一步更值得探索的是让分支利用群体层级、用户影响力等结构信息，并设置对应的欧氏对照；当前结果还不能说明这些结构已经被双曲分支学到了。


## 个人结论
当前“距离评分直接融合＋原损失从头联合训练”的设计存在严重问题，应该修改

其实从机制本身就存在重大的冲突：
- 我们希望距离分支学排序：正例近、负例远。
- 原 BCE 损失还要求校准绝对分数：大量负例的分数要低。
- 新增的负距离项恰好提供了一个整体压低分数的通道。即使正负例一起变远、排序没有改善，BCE 也可能下降。
- 联合训练时，这条优化路径还能改变主干表示，干扰原本应该学到的预测能力。
  
实验与这个解释是一致的：欧氏和双曲一起退化，说明共同的评分与优化设计有问题；冻结主干后，原融合损失也能正常工作，说明保护主干能避免这种破坏性影响。

下一步感觉可以考虑**改一下训练目标**试试？比如原 GRID 评分继续承担原损失，距离分支单独用 BPR，避免让距离分支通过整体压低融合分数来优化 BCE。 

可以再比较允许与阻断分支梯度回传主干的效果，判断是否还需要隔离梯度。




# 20261027
## 损失分开的实验
### 1. 欧氏分支：

运行命令：
```bash
python -u main.py --dataset twitter --decoder_type euclidean \
  --decoder_objective separate --decoder_loss_weight 0.1 --decoder_weight 0.1 \
  --decoder_detach_backbone --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_detach.log
```

结果：
-------Final Results-------
 recall@50: 0.25926033
 recall@100: 0.29954690
 ndcg@50: 0.19614133
 ndcg@100: 0.20963079

### 2. 双曲分支：

运行命令：
```bash
python -u main.py --dataset twitter --decoder_type hyperbolic \
  --decoder_objective separate --decoder_loss_weight 0.1 --decoder_weight 0.1 \
  --decoder_detach_backbone --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_detach.log
```

结果：
-------Final Results-------
 recall@50: 0.26024395
 recall@100: 0.29975054
 ndcg@50: 0.19639383
 ndcg@100: 0.20979795


可以看到结果：
1. 无需预训练和冻结主干，也能恢复正常性能。 这次主干从头训练，仅将损失拆开并隔离分支梯度，就从此前 Recall@50 约 0.0457 恢复到 0.26。这说明此前严重退化与评分融合后的训练目标及梯度耦合有关。

2. 双曲提供了一定的正向信号，但当前收益主要仍来自辅助评分。 双曲相对基线的 Recall@50、NDCG@50 分别提升约 1.51%、1.02%；相对欧氏则只提升约 0.38%、0.13%，几乎可以忽略不记。可以认为双曲在当前方案下略有优势，但并没有表现出明显的结构性优势。

3. 当前验证的是“独立损失＋梯度隔离”的整体效果。 两条命令都开启了禁止距离分支的梯度回传 --decoder_detach_backbone，因此尚不能把恢复完全归功于拆分损失。下一步是保持其他配置不变，去掉这个开关：若仍正常，说明拆分损失已足以避免退化；若明显下降，则说明分支 BPR 回传主干仍存在干扰。



# 20261028
## 独立损失，允许分支梯度回传主干的实验
### 1. 欧氏分支：
运行指令：
```bash
# 欧氏：独立损失，允许分支梯度回传主干
python -u main.py --dataset twitter --decoder_type euclidean \
  --decoder_objective separate --decoder_loss_weight 0.1 --decoder_weight 0.1 \
  --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_euclidean_joint.log
```

结果：
-------Final Results-------
 recall@50: 0.21095744
 recall@100: 0.26164433
 ndcg@50: 0.15432604
 ndcg@100: 0.17063250


### 2. 双曲分支：
#### （1）0.1融合
运行指令：
```bash
# 双曲：独立损失，允许分支梯度回传主干
python -u main.py --dataset twitter --decoder_type hyperbolic \
  --decoder_objective separate --decoder_loss_weight 0.1 --decoder_weight 0.1 \
  --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_hyperbolic_joint.log
```

结果：
-------Final Results-------
 recall@50: 0.21462515
 recall@100: 0.26589060
 ndcg@50: 0.15485239
 ndcg@100: 0.17133096

#### （2）0.05融合
运行指令：
```bash
# 双曲：独立损失，允许分支梯度回传主干
python -u main.py --dataset twitter --decoder_type hyperbolic \
  --decoder_objective separate --decoder_loss_weight 0.05 --decoder_weight 0.05 \
  --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_hyperbolic_joint.log
```

结果：
-------Final Results-------
 recall@50: 0.22645959
 recall@100: 0.27482146
 ndcg@50: 0.17084217
 ndcg@100: 0.18653291

#### （3）0.01融合
运行指令：
```bash
# 双曲：独立损失，允许分支梯度回传主干
python -u main.py --dataset twitter --decoder_type hyperbolic \
  --decoder_objective separate --decoder_loss_weight 0.01 --decoder_weight 0.01 \
  --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_hyperbolic_joint.log
```

结果：
-------Final Results-------
 recall@50: 0.24786288
 recall@100: 0.28757146
 ndcg@50: 0.18837656
 ndcg@100: 0.20193994

#### （4）0融合
运行指令：
```bash
# 双曲：独立损失，允许分支梯度回传主干
python -u main.py --dataset twitter --decoder_type hyperbolic \
  --decoder_objective separate --decoder_loss_weight 0 --decoder_weight 0 \
  --selection_split val --deterministic_eval \
  --early_stop_after_warmup --decoder_diagnostics --compare_decoder \
  2>&1 | tee result_record/twitter_separate_hyperbolic_joint.log
```

结果：
-------Final Results-------
 recall@50: 0.25538608
 recall@100: 0.29918081
 ndcg@50: 0.19420218
 ndcg@100: 0.20828202