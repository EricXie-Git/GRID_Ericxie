# This is GRID modified by EricXie

import os
import time
import logging
import json

import torch
import torch.nn.functional as F

from utils.Setup import setup
from utils.Optim import build_optimizer
from utils.Metric import Metrics
from dataLoader import create_dataloaders, load_social_graph, batch_process
from config import parse_args

from model import GRID
from module import Combined_Loss, BPR_Loss
from utils.DecoderExperiment import snapshot_state, save_checkpoint, load_checkpoint, DecoderDiagnostics, separate_loss


def train(args):
    """Full training loop with early stopping and best-model tracking."""
    global best_scores
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)
    graph = load_social_graph(args)

    model = GRID(args)
    model = model.to(args.device)
    if getattr(args, 'load_checkpoint', None):
        load_checkpoint(args.load_checkpoint, model, args)
    if getattr(args, 'freeze_backbone', False):
        model.freeze_backbone()

    diagnostics_enabled = getattr(args, 'decoder_diagnostics', False)
    decoder_bpr = getattr(args, 'decoder_objective', 'fused') == 'bpr'
    decoder_separate = getattr(args, 'decoder_objective', 'fused') == 'separate'
    compare = getattr(args, 'compare_decoder', False)
    selection_split = getattr(args, 'selection_split', 'test')
    selection_loader = val_dataloader if selection_split == 'val' else test_dataloader
    if getattr(args, 'freeze_backbone', False) and compare:
        on, off = inference(args, model, selection_loader, graph, compare_decoder=True)
        logging.info('Initial frozen %s ON=%s OFF=%s', selection_split, on, off)

    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)

    step = 0
    best_score = args.best_score
    start_time = time.time()

    patience = getattr(args, 'patience', 5)
    epochs_without_improvement = 0
    best_model_state = None
    best_scores = {}

    for epoch in range(args.max_epochs):
        print(f'\n[ Training Epoch {epoch} ]')

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        diagnostics = DecoderDiagnostics() if diagnostics_enabled else None

        for batch in train_dataloader:
            cascade, cas_mask, labels_padded, label_mask, negative_samples, neg_mask, first_label, previous_mask = batch_process(args, batch)
            need_details = diagnostics_enabled or decoder_bpr or decoder_separate
            output = model(args, cascade, cas_mask, previous_mask, graph, return_details=need_details)
            pred_users, details = output if need_details else (output, None)
            loss_inputs = (args, pred_users, labels_padded, label_mask, negative_samples, neg_mask, first_label)
            components = {}
            if decoder_separate:
                loss, components = separate_loss(args, details, previous_mask, labels_padded,
                                                 label_mask, negative_samples, neg_mask, first_label)
            elif decoder_bpr:
                # 单独训练未乘融合权重的 Decoder 排序，不让 BCE 影响 Decoder。
                loss = BPR_Loss(args, details['decoder_scores'], labels_padded, label_mask,
                                negative_samples, neg_mask)
                if diagnostics_enabled:
                    with torch.no_grad():
                        _, components = Combined_Loss(*loss_inputs, return_components=True)
            elif diagnostics_enabled:
                loss, components = Combined_Loss(*loss_inputs, return_components=True)
            else:
                loss = Combined_Loss(*loss_inputs)
            if diagnostics is not None:
                diagnostics.update(args, details, previous_mask, labels_padded, label_mask,
                                   negative_samples, neg_mask, components, loss)

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

        if diagnostics is not None:
            logging.info('Epoch %s diagnostics (%s): %s', epoch, getattr(args, 'decoder_objective', 'fused'),
                         json.dumps(diagnostics.summary(), sort_keys=True))
        if compare:
            t_scores, off_scores = inference(args, model, selection_loader, graph, compare_decoder=True)
            logging.info('Epoch %s %s ON=%s OFF=%s', epoch, selection_split, t_scores, off_scores)
        else:
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
        if compare:
            best_scores, off_scores = inference(args, model, test_dataloader, graph, compare_decoder=True)
            logging.info('Final test ON=%s OFF=%s', best_scores, off_scores)
        else:
            best_scores = inference(args, model, test_dataloader, graph)

    print('\n #-------Final Results-------')
    for metric, value in best_scores.items():
        print(f' {metric}: {value:.8f}')


def inference(args, model, dataloader, graph, compare_decoder=False):
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
    off_sums = {k: torch.zeros_like(v) for k, v in score_sums.items()} if compare_decoder else None

    with torch.no_grad():
        for batch in dataloader:
            cascade, cas_mask, labels, _, _, _, first_label, previous_mask = batch_process(args, batch)
            output = model(args, cascade, cas_mask, previous_mask, graph, return_details=compare_decoder)
            if compare_decoder:
                pred_users, details = output
                off_scores, _ = metrics.compute_recall_ndcg_multi_label(details['base_scores'] + previous_mask, labels)
                for name, value in off_scores.items():
                    off_sums[name] += value
            else:
                pred_users = output

            batch_scores, batch_size = metrics.compute_recall_ndcg_multi_label(pred_users, labels)

            for metric_name, metric_sum in batch_scores.items():
                score_sums[metric_name] += metric_sum

            total_samples += batch_size

    result = {k: (v / max(1, total_samples)).item() for k, v in score_sums.items()}
    if compare_decoder:
        return result, {k: (v / max(1, total_samples)).item() for k, v in off_sums.items()}
    return result

def main():
    args = parse_args()
    setup(args)

    os.makedirs(args.saved_model_path, exist_ok=True)
    logging.info("Training/Testing parameters: %s", args)
    train(args)


if __name__ == '__main__':
    main()
