import json
import random
import numpy as np

import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from utils.Setup import trans_to_cuda


def create_cascade_collate_fn(max_len):
    """Return a collate function that pads cascades and negative samples to fixed lengths."""
    def cascade_collate_fn(batch):
        batch_size = len(batch)

        max_neg_len = max(len(item['negative_samples']) for item in batch)
        if max_neg_len == 0:
            max_neg_len = 1

        observed_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        negative_samples_padded = torch.zeros(batch_size, max_neg_len, dtype=torch.long)

        labels_list = []
        label_counts = []
        neg_counts = []

        for i, item in enumerate(batch):
            observed_tensor = torch.tensor(item['observed'], dtype=torch.long)
            if len(observed_tensor) > max_len:
                observed_tensor = observed_tensor[:max_len]
            observed_padded[i, :len(observed_tensor)] = observed_tensor


            neg_tensor = torch.tensor(item['negative_samples'], dtype=torch.long) if item['negative_samples'] else torch.tensor([0], dtype=torch.long)
            actual_neg_len = min(len(neg_tensor), max_neg_len)
            negative_samples_padded[i, :actual_neg_len] = neg_tensor[:actual_neg_len]

            labels_list.append(item['labels'])
            label_counts.append(item['label_count'])
            neg_counts.append(max(1, len(item['negative_samples'])))

        labels = torch.stack(labels_list)
        label_counts = torch.tensor(label_counts, dtype=torch.long)
        neg_counts = torch.tensor(neg_counts, dtype=torch.long)

        return {
            'observed': observed_padded,
            'labels': labels,
            'label_count': label_counts,
            'negative_samples': negative_samples_padded,
            'neg_count': neg_counts
        }

    return cascade_collate_fn


def create_dataloaders(args):
    """Build train/val/test DataLoaders from cascade data files."""
    train_dataset = CascadeData(args, args.cascade_path_train)
    val_dataset = CascadeData(args, args.cascade_path_valid)
    test_dataset = CascadeData(args, args.cascade_path_test)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=create_cascade_collate_fn(args.max_len),
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                collate_fn=create_cascade_collate_fn(args.max_len),
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 collate_fn=create_cascade_collate_fn(args.max_len),
                                 pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader


class CascadeData(Dataset):
    """Dataset wrapping cascade JSON files with on-the-fly negative sampling."""

    def __init__(self, args, dataPath):
        self.max_len = args.max_len
        self.EOS = args.user_num - 1
        self.seed = args.seed
        self.neg_num = args.neg_num

        with open(dataPath, 'r') as cas_file:
            self.cascade_data = json.load(cas_file)

    def __len__(self) -> int:
        return len(self.cascade_data)

    def __getitem__(self, idx: int) -> dict:
        observed_cascade = self.cascade_data[idx]['observed']
        labels = self.cascade_data[idx]['label']

        all_negative_samples = self.cascade_data[idx].get('neg', [])

        if all_negative_samples:
            if self.seed is not None:
                random.seed(self.seed + idx)
            num_to_sample = min(self.neg_num, len(all_negative_samples))
            negative_samples = random.sample(all_negative_samples, num_to_sample)
        else:
            negative_samples = []

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        labels_padded = pad_tensor(labels_tensor, 20, pad_value=0)

        data = dict(
            observed=observed_cascade,
            labels=labels_padded,
            negative_samples=negative_samples,
            label_count=len(labels)
        )
        return data

    def get_labels(self, idx: int) -> List[int]:
        """Return the ground-truth label list for cascade at index ``idx``."""
        return self.cascade_data[idx]['label']


def batch_process(args, data):
    """Transfer a collated batch to the target device and build auxiliary masks."""
    observed_pad = data['observed']
    previous_mask = trans_to_cuda(get_previous_user_mask(observed_pad, args.user_num))

    device_tensors = {
        'cascade': observed_pad.long(),
        'cas_mask': (observed_pad == 0),
        'labels_padded': data['labels'],
        'negative_samples': data['negative_samples']
    }

    for key in device_tensors:
        device_tensors[key] = trans_to_cuda(device_tensors[key])

    cascade = device_tensors['cascade']
    cas_mask = device_tensors['cas_mask']
    labels_padded = device_tensors['labels_padded']
    negative_samples = device_tensors['negative_samples']

    label_count = trans_to_cuda(data['label_count'])
    neg_count = trans_to_cuda(data['neg_count'])

    batch_size, max_labels = labels_padded.shape
    label_indices = torch.arange(max_labels, device=labels_padded.device).unsqueeze(0).expand(batch_size, -1)
    label_mask = label_indices < label_count.unsqueeze(1)

    batch_size, max_neg = negative_samples.shape
    neg_indices = torch.arange(max_neg, device=negative_samples.device).unsqueeze(0).expand(batch_size, -1)
    neg_mask = neg_indices < neg_count.unsqueeze(1)

    first_label = labels_padded[:, 0] if labels_padded.size(1) > 0 else torch.zeros(batch_size, dtype=torch.long, device=labels_padded.device)

    return cascade, cas_mask, labels_padded, label_mask, negative_samples, neg_mask, first_label, previous_mask


def load_social_graph(args) -> Optional[Data]:
    """Load the pre-built social graph from disk and move it to the target device."""
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_path}\n"
            "Run `python scripts/convert_graph_to_pt.py` to generate it from graph.txt."
        )
    graph_data: Data = torch.load(graph_path, weights_only=False)
    return graph_data.to(args.device)


def load_embeddings(file_path: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    embeddings = np.loadtxt(file_path)[:, 1:]
    return torch.tensor(embeddings, dtype=torch.float32, device=device)


def pad_tensor(tensor, max_len, pad_value=0):
    """Pad or truncate ``tensor`` along dim 0 to length ``max_len``."""
    if tensor.dim() == 1:
        len_seq = tensor.size(0)
        if len_seq >= max_len:
            return tensor[:max_len]
        out = tensor.new_full((max_len,), pad_value)
        out[:len_seq] = tensor
        return out

    if tensor.dim() == 2:
        tensor = tensor.long()
        len_seq, _feature_dim = tensor.shape
        if len_seq >= max_len:
            return tensor[:max_len, :]
        out = tensor.new_full((max_len, tensor.size(1)), pad_value)
        out[:len_seq] = tensor
        return out

    raise ValueError("Only 1D or 2D tensors are supported.")


def get_previous_user_mask(seq: torch.Tensor, user_size: int) -> torch.Tensor:
    """Build a logit mask that suppresses re-prediction of users already in the cascade."""
    if seq.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {seq.dim()}D")

    batch_size, _ = seq.size()

    float_mask = torch.zeros(batch_size, user_size, dtype=torch.float32, device=seq.device)

    valid_mask = (seq > 0) & (seq < user_size)
    batch_indices, seq_indices = torch.where(valid_mask)
    float_mask[batch_indices, seq[batch_indices, seq_indices]] = float('-1e9')

    return float_mask
