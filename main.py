# This is GRID modified by EricXie

import os
import time
import logging

import torch
import torch.nn.functional as F

from utils.Setup import setup
from utils.Optim import build_optimizer
from utils.Metric import Metrics
from dataLoader import create_dataloaders, load_social_graph, batch_process
from config import parse_args

from model import GRID
from module import Combined_Loss
from utils.Checkpoint import snapshot_state, save_checkpoint, load_checkpoint


def train(args):
    """Full training loop with early stopping and best-model tracking."""
    global best_scores
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)
    graph = load_social_graph(args)

    model = GRID(args)
    model = model.to(args.device)
    if getattr(args, 'load_checkpoint', None):
        load_checkpoint(args.load_checkpoint, model, args)

    selection_split = getattr(args, 'selection_split', 'test')
    selection_loader = val_dataloader if selection_split == 'val' else test_dataloader
    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)

    step = 0
    best_score = args.best_score
    start_time = time.time()

    patience = getattr(args, 'patience', 5)
    epochs_without_improvement = 0
    best_model_state = None
    best_scores = {}

    sep_weight = getattr(args, 'group_sep_weight', 0.0)
    pref_weight = getattr(args, 'group_pref_weight', 0.0)
    use_structure = bool(sep_weight or pref_weight)
    for epoch in range(args.max_epochs):
        print(f'\n[ Training Epoch {epoch} ]')

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            cascade, cas_mask, labels_padded, label_mask, negative_samples, neg_mask, first_label, previous_mask = batch_process(args, batch)
            output = model(args, cascade, cas_mask, previous_mask, graph,
                           first_label=first_label if model.group_loss_weight > 0 else None,
                           return_structure=use_structure)
            if use_structure:
                output, (sep_loss, pref_loss) = output
            if model.group_loss_weight > 0:
                pred_users, logits, target, _ = output
            else:
                pred_users = output
            loss = Combined_Loss(args, pred_users, labels_padded, label_mask, negative_samples, neg_mask, first_label)

            if model.group_loss_weight > 0:
                # 按有效样本数归一化，全被忽略时辅助损失为零。
                group_loss = F.cross_entropy(logits, target, reduction='sum') / (target != -100).sum().clamp_min(1)
                loss = loss + model.group_loss_weight * group_loss

            if use_structure:
                loss = loss + sep_weight * sep_loss + pref_weight * pref_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                avg_loss = epoch_loss / num_batches
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {avg_loss:.3f}")

        t_scores = inference(args, model, selection_loader, graph)

        print(f' # ----------{selection_split.capitalize()} Result---------')
        for metric, value in t_scores.items():
            print(f' {metric}: {value:.8f}')

        current_score = sum(t_scores.values())

        if best_model_state is None or current_score > best_score:
            best_score = current_score
            best_scores = t_scores.copy()
            best_model_state = snapshot_state(model)
            if getattr(args, 'save_checkpoint', False):
                save_checkpoint(args.ckpt_file, best_model_state, args, epoch, best_scores)
            epochs_without_improvement = 0
            print(' --> Save Model <-- ')
        else:
            if not getattr(args, 'early_stop_after_warmup', False) or step >= args.n_warmup_steps:
                epochs_without_improvement += 1
            print(f' --> No improvement for {epochs_without_improvement} epochs <-- ')

        if epochs_without_improvement >= patience:
            print(f'\n --> Early stopping triggered after {patience} epochs without improvement <-- ')
            print(f' --> Best score achieved: {best_score:.8f} <-- ')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(' --> Loaded best model state <-- ')

    # 验证集只用于选择模型；选定后才报告一次测试集结果。
    if selection_split == 'val':
        best_scores = inference(args, model, test_dataloader, graph)

    print('\n #-------Final Results-------')
    for metric, value in best_scores.items():
        print(f' {metric}: {value:.8f}')


def inference(args, model, dataloader, graph):
    """Run evaluation and return averaged Recall@K and NDCG@K scores."""
    model.eval()
    metrics = Metrics(args)
    device = args.device

    score_sums = {
        f'recall@{k}': torch.zeros(1, device=device)
        for k in args.metric_k
    }
    score_sums.update({
        f'ndcg@{k}': torch.zeros(1, device=device)
        for k in args.metric_k
    })

    group_hits = torch.zeros(3, device=device, dtype=torch.long) if model.group_loss_weight > 0 else None
    grouping_stats = None
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            cascade, cas_mask, labels, _, _, _, first_label, previous_mask = batch_process(args, batch)
            if group_hits is not None:
                pred_users, logits, target, largest = model(args, cascade, cas_mask, previous_mask, graph, first_label)
                group_hits += torch.stack(((logits.argmax(dim=1) == target).sum(),
                                           (largest == target).sum(), (target != -100).sum()))
            else:
                pred_users = model(args, cascade, cas_mask, previous_mask, graph)

            batch_scores, batch_size = metrics.compute_recall_ndcg_multi_label(pred_users, labels)

            for metric_name, metric_sum in batch_scores.items():
                score_sums[metric_name] += metric_sum

            stats = getattr(getattr(model, 'Grouping', None), 'eval_stats', None)
            if stats is not None:
                grouping_stats = stats if grouping_stats is None else grouping_stats + stats
            total_samples += batch_size

    if grouping_stats is not None:
        occupied, cascades, confidence, radius, tokens, *diagnostics = grouping_stats.tolist()
        print(f' Grouping: occupied/cascade={occupied / max(1, cascades):.2f}, '
              f'mean_max_assignment={confidence / max(1, tokens):.4f}, mean_radius={radius / max(1, tokens):.4f}')
        if diagnostics:
            first_p, last_p, first_dist, first_pairs, last_dist, last_pairs, *pref = diagnostics
            print(f' Grouping first/final: max_assignment={first_p/max(1, tokens):.5f}/{last_p/max(1, tokens):.5f}, '
                  f'center_distance={first_dist/max(1, first_pairs):.5f}/{last_dist/max(1, last_pairs):.5f}')
            if pref:
                reached, eligible = pref
                print(f' Group preference: ratio_target_met={int(reached)}/{int(eligible)}, '
                      f'target={model.Grouping.structure.ratio:g}')
    if group_hits is not None:
        correct, baseline, valid = group_hits.tolist()
        print(f' Next group: accuracy={correct / valid if valid else float("nan"):.4f}, '
              f'largest_group_accuracy={baseline / valid if valid else float("nan"):.4f}, valid={valid}/{total_samples}')
    result = {k: (v / max(1, total_samples)).item() for k, v in score_sums.items()}
    return result

def main():
    args = parse_args()
    setup(args)

    os.makedirs(args.saved_model_path, exist_ok=True)
    logging.info("Training/Testing parameters: %s", args)
    train(args)


if __name__ == '__main__':
    main()
