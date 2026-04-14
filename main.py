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


def train(args):
    """Full training loop with early stopping and best-model tracking."""
    global best_scores
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)
    graph = load_social_graph(args)

    model = GRID(args)
    model = model.to(args.device)

    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)

    step = 0
    best_score = args.best_score
    start_time = time.time()

    patience = 5
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(args.max_epochs):
        print(f'\n[ Training Epoch {epoch} ]')

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            cascade, cas_mask, labels_padded, label_mask, negative_samples, neg_mask, first_label, previous_mask = batch_process(args, batch)
            pred_users = model(args, cascade, cas_mask, previous_mask, graph)

            loss = Combined_Loss(args, pred_users, labels_padded, label_mask, negative_samples, neg_mask, first_label)

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

        t_scores = inference(args, model, test_dataloader, graph)

        print(' # ----------Test Result---------')
        for metric, value in t_scores.items():
            print(f' {metric}: {value:.8f}')

        current_score = sum(t_scores.values())

        if current_score > best_score:
            best_score = current_score
            best_scores = t_scores.copy()
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(' --> Save Model <-- ')
        else:
            epochs_without_improvement += 1
            print(f' --> No improvement for {epochs_without_improvement} epochs <-- ')

        if epochs_without_improvement >= patience:
            print(f'\n --> Early stopping triggered after {patience} epochs without improvement <-- ')
            print(f' --> Best score achieved: {best_score:.8f} <-- ')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(' --> Loaded best model state <-- ')

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

    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            cascade, cas_mask, labels, _, _, _, first_label, previous_mask = batch_process(args, batch)
            pred_users = model(args, cascade, cas_mask, previous_mask, graph)

            batch_scores, batch_size = metrics.compute_recall_ndcg_multi_label(pred_users, labels)

            for metric_name, metric_sum in batch_scores.items():
                score_sums[metric_name] += metric_sum

            total_samples += batch_size

    if total_samples > 0:
        return {k: (v / total_samples).item() for k, v in score_sums.items()}
    return {k: 0.0 for k in score_sums}

def main():
    args = parse_args()
    setup(args)

    os.makedirs(args.saved_model_path, exist_ok=True)
    logging.info("Training/Testing parameters: %s", args)
    train(args)


if __name__ == '__main__':
    main()
