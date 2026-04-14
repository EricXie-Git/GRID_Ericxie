import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, min_lr=1e-6, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_lr = 0.5 * (1.0 + math.cos(math.pi * num_cycles * progress))
        return max(min_lr, cosine_lr)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_optimizer(args, model, num_training_steps):
    """ Prepare optimizer and schedule """
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2)
    scheduler = schedule_with_warmup(optimizer, args.n_warmup_steps, num_training_steps)

    return optimizer, scheduler
