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
