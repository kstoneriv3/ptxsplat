from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import numpy as np
import torch

from ._common import environment_metadata, require_cuda, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure lightweight launch, DRAM-copy, and FP32 GEMM ceilings."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--launch-iterations", type=int, default=1000)
    parser.add_argument(
        "--dram-bytes",
        type=int,
        default=64 * 1024**2,
        help="Bytes per source/destination allocation (default: 64 MiB).",
    )
    parser.add_argument("--gemm-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout.")
    return parser


def _event_times(
    operation: Callable[[], Any], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _stats(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "minimum_ms": float(np.min(values)),
        "q1_ms": float(np.percentile(values, 25.0)),
        "q3_ms": float(np.percentile(values, 75.0)),
    }


def _launch_probe(iterations: int) -> dict[str, Any]:
    value = torch.zeros((), device="cuda")
    for _ in range(100):
        value.add_(1.0)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    host_start = time.perf_counter_ns()
    for _ in range(iterations):
        value.add_(1.0)
    host_end = time.perf_counter_ns()
    end_event.record()
    end_event.synchronize()
    return {
        "operation": "one-element in-place add",
        "iterations": iterations,
        "host_enqueue_us_per_kernel": (host_end - host_start) / iterations / 1e3,
        "gpu_serialized_us_per_kernel": start_event.elapsed_time(end_event)
        * 1e3
        / iterations,
    }


def _dram_probe(num_bytes: int, warmup: int, iterations: int) -> dict[str, Any]:
    elements = num_bytes // torch.empty((), dtype=torch.float32).element_size()
    source = torch.rand(elements, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    samples = _event_times(lambda: destination.copy_(source), warmup, iterations)
    statistics = _stats(samples)
    transferred_bytes = elements * 4 * 2
    statistics.update(
        {
            "operation": "device-to-device float32 copy",
            "allocation_bytes_each": elements * 4,
            "bytes_per_iteration": transferred_bytes,
            "bandwidth_gb_s_at_median": transferred_bytes
            / (statistics["median_ms"] * 1e-3)
            / 1e9,
            "bandwidth_gb_s_at_minimum": transferred_bytes
            / (statistics["minimum_ms"] * 1e-3)
            / 1e9,
        }
    )
    return statistics


def _fp32_probe(size: int, warmup: int, iterations: int) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    left = torch.randn((size, size), device="cuda", dtype=torch.float32)
    right = torch.randn((size, size), device="cuda", dtype=torch.float32)
    output = torch.empty_like(left)
    samples = _event_times(
        lambda: torch.mm(left, right, out=output), warmup, iterations
    )
    statistics = _stats(samples)
    flops = 2 * size**3
    statistics.update(
        {
            "operation": "IEEE float32 GEMM with TF32 disabled",
            "matrix_size": size,
            "flops_per_iteration": flops,
            "throughput_tflops_at_median": flops
            / (statistics["median_ms"] * 1e-3)
            / 1e12,
            "throughput_tflops_at_minimum": flops
            / (statistics["minimum_ms"] * 1e-3)
            / 1e12,
        }
    )
    return statistics


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.launch_iterations <= 0
        or args.dram_bytes < 4
        or args.gemm_size <= 0
    ):
        parser.error("counts and sizes must be positive; warmup may be zero")
    require_cuda()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = {
        "schema_version": 1,
        "benchmark": "roofline-probes",
        "seed": args.seed,
        "environment": environment_metadata(),
        "probes": {
            "launch": _launch_probe(args.launch_iterations),
            "dram": _dram_probe(args.dram_bytes, args.warmup, args.iterations),
            "fp32": _fp32_probe(args.gemm_size, args.warmup, args.iterations),
        },
    }
    write_json(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
