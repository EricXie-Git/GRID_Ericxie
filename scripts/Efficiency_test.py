import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from utils.Setup import setup
from dataLoader import create_dataloaders, load_social_graph, batch_process
from config import parse_args
from model import GRID


class _ModelWrapper(torch.nn.Module):
    """Wraps GRID to expose only tensor inputs for FLOPs profiling."""

    def __init__(self, model, args, graph):
        super().__init__()
        self.model = model
        self.args = args
        self.graph = graph

    def forward(self, cascade, cas_mask, previous_mask):
        return self.model(self.args, cascade, cas_mask, previous_mask, self.graph)


def _measure_complexity(model, args, graph, sample_batch):
    """Return (flops_or_None, param_count)."""
    param_count = sum(p.numel() for p in model.parameters())

    if not THOP_AVAILABLE:
        return None, param_count

    try:
        cascade, cas_mask, *_, previous_mask = batch_process(args, sample_batch)
        wrapper = _ModelWrapper(model, args, graph).eval()
        with torch.no_grad():
            flops, _ = profile(
                wrapper,
                inputs=(cascade[:1], cas_mask[:1], previous_mask[:1]),
                verbose=False,
            )
        return flops, param_count
    except Exception as e:
        print(f"  [warn] FLOPs profiling failed: {e}")
        return None, param_count


def _measure_inference_time(model, args, graph, dataloader, num_batches=50):
    """Return per-sample latencies (ms) measured with CUDA Events on GPU or perf_counter on CPU."""
    use_cuda = torch.cuda.is_available()
    times_ms = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            cascade, cas_mask, *_, previous_mask = batch_process(args, batch)
            batch_size = cascade.size(0)

            if use_cuda:
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                model(args, cascade, cas_mask, previous_mask, graph)
                t1.record()
                torch.cuda.synchronize()
                elapsed_ms = t0.elapsed_time(t1)
            else:
                ts = time.perf_counter()
                model(args, cascade, cas_mask, previous_mask, graph)
                elapsed_ms = (time.perf_counter() - ts) * 1000.0

            times_ms.append(elapsed_ms / batch_size)

    return np.array(times_ms)


def efficiency_test(args):
    """Profile model parameters, FLOPs, GPU memory, and per-sample inference latency."""
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  GRID Efficiency Test  |  dataset={args.dataset}  |  device={args.device}")
    print(sep)

    print("\n[1/3] Loading data and model...")
    _, _, test_dataloader = create_dataloaders(args)
    graph = load_social_graph(args)

    model = GRID(args).to(args.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_after_load = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0

    print("[2/3] Profiling model complexity...")
    sample_batch = next(iter(test_dataloader))
    flops, param_count = _measure_complexity(model, args, graph, sample_batch)

    print("[3/3] Warming up and timing inference...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= 3:
                break
            cascade, cas_mask, *_, previous_mask = batch_process(args, batch)
            model(args, cascade, cas_mask, previous_mask, graph)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms = _measure_inference_time(model, args, graph, test_dataloader, num_batches=50)
    mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0

    mean_ms = float(np.mean(times_ms))

    print(f"\n{sep}")
    print("  RESULTS")
    print(sep)

    print(f"\n  Model Complexity")
    print(f"    Parameters : {param_count:,}")
    if flops is not None:
        flops_str, params_str = clever_format([flops, param_count], "%.3f")
        print(f"    FLOPs      : {flops_str}  (params: {params_str})")
    else:
        print(f"    FLOPs      : n/a (install thop for FLOPs counting)")

    print(f"\n  GPU Memory")
    print(f"    After load : {mem_after_load:.3f} GB")
    print(f"    Peak       : {mem_peak:.3f} GB")

    print(f"\n  Inference Time  (per sample, {len(times_ms)} batches)")
    print(f"    Mean       : {mean_ms:.4f} ms")
    print(f"    Std        : {float(np.std(times_ms)):.4f} ms")
    print(f"    Throughput : {1000.0 / mean_ms:.1f} samples/sec")

    print(f"\n{sep}\n")

    return {
        'param_count': param_count,
        'flops': flops,
        'mem_after_load_gb': mem_after_load,
        'mem_peak_gb': mem_peak,
        'inference_mean_ms': mean_ms,
        'inference_std_ms': float(np.std(times_ms)),
    }


def main():
    args = parse_args()
    setup(args)
    results = efficiency_test(args)

if __name__ == '__main__':
    main()
