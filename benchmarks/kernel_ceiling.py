from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence


SCHEMA_VERSION = 2
SEED = 42
WIDTH = 1920
HEIGHT = 1080
TILE_SIZE = 16
GRID_BLOCKS = math.ceil(WIDTH / TILE_SIZE) * math.ceil(HEIGHT / TILE_SIZE)
THREADS = 256
WARPS_PER_CTA = THREADS // 32
DEFAULT_OUTPUT = Path("benchmark-results/kernel-ceiling")


CPP_SOURCE = r"""
#include <torch/extension.h>

void run_kernel_ceiling_probe_cuda(
    torch::Tensor output,
    int64_t kind,
    int64_t blocks,
    int64_t loops,
    int64_t active_warps,
    int64_t contention_ctas,
    int64_t skew_iterations,
    int64_t dynamic_smem);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_probe", &run_kernel_ceiling_probe_cuda);
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define KC_CHECK(call) do {                                               \
  cudaError_t error = (call);                                             \
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));           \
} while (0)

__device__ __forceinline__ float kc_ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float kc_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

extern "C" __global__ __launch_bounds__(256) void kc_reduction_ilp9(
    float *output, int loops) {
  float x[9];
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    x[k] = 0.00001f * float((threadIdx.x & 31) + k + 1);
  }
  for (int r = 0; r < loops; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int k = 0; k < 9; ++k) {
        x[k] += __shfl_xor_sync(0xffffffffu, x[k], offset);
      }
    }
#pragma unroll
    for (int k = 0; k < 9; ++k) x[k] *= 0.03125f;
  }
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 9; ++k) sum += x[k];
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

extern "C" __global__ __launch_bounds__(256) void kc_reduction_chain(
    float *output, int loops) {
  float x = 0.00001f * float((threadIdx.x & 31) + 1);
  for (int r = 0; r < loops; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      x += __shfl_xor_sync(0xffffffffu, x, offset);
    }
    x *= 0.03125f;
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

extern "C" __global__ __launch_bounds__(256) void kc_redux_max_chain(
    float *output, int loops) {
  int x = (threadIdx.x & 31) + 1;
  for (int r = 0; r < loops; ++r) {
    x = __reduce_max_sync(0xffffffffu, x + (r & 7));
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = float(x);
}

extern "C" __global__ __launch_bounds__(256) void kc_atomic_redg(
    float *output, int loops, int active_warps, int contention_ctas) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0 && warp < active_warps) {
    const int group = blockIdx.x / contention_ctas;
    float *address = output + group * 9;
    for (int r = 0; r < loops; ++r) {
#pragma unroll
      for (int k = 0; k < 9; ++k) atomicAdd(address + k, float(warp + 1));
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void kc_barrier(
    float *output, int loops, int skew_iterations) {
  float value = 0.0001f * float(threadIdx.x + 1);
  for (int r = 0; r < loops; ++r) {
    if ((threadIdx.x & 31) < 16) {
      for (int k = 0; k < skew_iterations; ++k) {
        value = fmaf(value, 1.000001f, 0.000001f);
      }
    }
    __syncthreads();
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

extern "C" __global__ __launch_bounds__(256) void kc_shared_lds128(
    float *output, int loops) {
  extern __shared__ float4 stage[];
  for (int i = threadIdx.x; i < 768; i += blockDim.x) {
    stage[i] = make_float4(float(i), float(i + 1), float(i + 2), float(i + 3));
  }
  __syncthreads();
  float sum = 0.f;
  for (int r = 0; r < loops; ++r) {
    const int index = (r % 384) * 2;
    const unsigned address0 = static_cast<unsigned>(
        __cvta_generic_to_shared(stage + index));
    const unsigned address1 = address0 + sizeof(float4);
    float4 a, b;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w)
                 : "r"(address0));
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(b.x), "=f"(b.y), "=f"(b.z), "=f"(b.w)
                 : "r"(address1));
    sum += a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

extern "C" __global__ __launch_bounds__(256) void kc_shared_sts128(
    float *output, int loops) {
  extern __shared__ float4 stage[];
  const float4 a = make_float4(float(threadIdx.x), 2.f, 3.f, 4.f);
  const float4 b = make_float4(5.f, 6.f, 7.f, 8.f);
  for (int r = 0; r < loops; ++r) {
    const int index = ((threadIdx.x + r * 256) % 384) * 2;
    stage[index] = a;
    stage[index + 1] = b;
    asm volatile("" ::: "memory");
  }
  __syncthreads();
  output[blockIdx.x * blockDim.x + threadIdx.x] = stage[threadIdx.x].x;
}

extern "C" __global__ __launch_bounds__(256) void kc_mufu_ex2_ilp4(
    float *output, int loops) {
  const float x0 = -0.25f - 0.00001f * float(threadIdx.x & 31);
  const float x1 = x0 - 0.25f;
  const float x2 = x0 - 0.50f;
  const float x3 = x0 - 0.75f;
  float sum = 0.f;
  for (int r = 0; r < loops; ++r) {
    const float vary = 0.00001f * float(r & 31);
    sum += kc_ex2(x0 - vary) + kc_ex2(x1 - vary) +
           kc_ex2(x2 - vary) + kc_ex2(x3 - vary);
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

extern "C" __global__ __launch_bounds__(256) void kc_mufu_ex2_chain(
    float *output, int loops) {
  float x = -0.5f - 0.00001f * float(threadIdx.x & 31);
  for (int r = 0; r < loops; ++r) x = -1.0f + 0.5f * kc_ex2(x);
  output[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

extern "C" __global__ __launch_bounds__(256) void kc_mufu_rcp_ilp4(
    float *output, int loops) {
  const float x0 = 1.25f + 0.00001f * float(threadIdx.x & 31);
  const float x1 = x0 + 0.25f;
  const float x2 = x0 + 0.50f;
  const float x3 = x0 + 0.75f;
  float sum = 0.f;
  for (int r = 0; r < loops; ++r) {
    const float vary = 0.00001f * float(r & 31);
    sum += kc_rcp(x0 + vary) + kc_rcp(x1 + vary) +
           kc_rcp(x2 + vary) + kc_rcp(x3 + vary);
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

extern "C" __global__ __launch_bounds__(256) void kc_mufu_rcp_chain(
    float *output, int loops) {
  float x = 1.5f + 0.00001f * float(threadIdx.x & 31);
  for (int r = 0; r < loops; ++r) x = 1.0f + kc_rcp(x);
  output[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

extern "C" __global__ __launch_bounds__(256) void kc_empty_grid(float *output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) output[0] += 1.f;
}

template <typename Kernel>
void kc_set_smem(Kernel kernel, int dynamic_smem) {
  if (dynamic_smem > 49152) {
    KC_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem));
  }
}

void run_kernel_ceiling_probe_cuda(
    torch::Tensor output,
    int64_t kind,
    int64_t blocks,
    int64_t loops,
    int64_t active_warps,
    int64_t contention_ctas,
    int64_t skew_iterations,
    int64_t dynamic_smem) {
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat32);
  c10::cuda::CUDAGuard guard(output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned int>(blocks));
  dim3 block(256);
  switch (kind) {
    case 0:
      kc_set_smem(kc_reduction_ilp9, dynamic_smem);
      kc_reduction_ilp9<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 1:
      kc_set_smem(kc_reduction_chain, dynamic_smem);
      kc_reduction_chain<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 2:
      kc_set_smem(kc_atomic_redg, dynamic_smem);
      kc_atomic_redg<<<grid, block, dynamic_smem, stream>>>(
          output.data_ptr<float>(), loops, active_warps, contention_ctas);
      break;
    case 3:
      kc_set_smem(kc_barrier, dynamic_smem);
      kc_barrier<<<grid, block, dynamic_smem, stream>>>(
          output.data_ptr<float>(), loops, skew_iterations);
      break;
    case 4:
      kc_set_smem(kc_shared_lds128, dynamic_smem);
      kc_shared_lds128<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 5:
      kc_set_smem(kc_shared_sts128, dynamic_smem);
      kc_shared_sts128<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 6:
      kc_set_smem(kc_mufu_ex2_ilp4, dynamic_smem);
      kc_mufu_ex2_ilp4<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 7:
      kc_set_smem(kc_mufu_ex2_chain, dynamic_smem);
      kc_mufu_ex2_chain<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 8:
      kc_set_smem(kc_mufu_rcp_ilp4, dynamic_smem);
      kc_mufu_rcp_ilp4<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 9:
      kc_set_smem(kc_mufu_rcp_chain, dynamic_smem);
      kc_mufu_rcp_chain<<<grid, block, dynamic_smem, stream>>>(output.data_ptr<float>(), loops);
      break;
    case 10:
      kc_empty_grid<<<grid, block, 0, stream>>>(output.data_ptr<float>());
      break;
    case 11:
      kc_set_smem(kc_redux_max_chain, dynamic_smem);
      kc_redux_max_chain<<<grid, block, dynamic_smem, stream>>>(
          output.data_ptr<float>(), loops);
      break;
    default:
      TORCH_CHECK(false, "unknown kernel-ceiling probe kind");
  }
  KC_CHECK(cudaGetLastError());
}
"""


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    kind: int
    blocks_per_sm: int
    loops: int
    active_warps: int = 8
    contention_ctas: int = 1
    skew_iterations: int = 0
    dynamic_smem: int = 18_432
    operation: str = "operations"
    operations_per_cta_loop: int = 1
    operation_scale: int = 1

    def blocks(self, sm_count: int) -> int:
        if self.name == "launch_one_cta":
            return 1
        if self.name == "launch_full_grid":
            return GRID_BLOCKS
        return self.blocks_per_sm * sm_count

    def operations(self, sm_count: int) -> int:
        if self.kind == 10:
            return 1
        return (
            self.blocks(sm_count)
            * self.loops
            * self.operations_per_cta_loop
            * self.operation_scale
        )


PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec("reduction_ilp9", 0, 5, 128, operation="shuffle_add_pairs", operations_per_cta_loop=8 * 45),
    ProbeSpec("reduction_chain", 1, 5, 512, operation="five_stage_reductions", operations_per_cta_loop=8),
    ProbeSpec("redux_max_chain", 11, 5, 512, operation="redux_max_warp_instructions", operations_per_cta_loop=8),
    ProbeSpec("atomic_warp4", 2, 5, 64, active_warps=4, operation="redg_fp32", operations_per_cta_loop=4 * 9),
    ProbeSpec("atomic_warp8", 2, 5, 64, active_warps=8, operation="redg_fp32", operations_per_cta_loop=8 * 9),
    ProbeSpec("atomic_crosscta8", 2, 5, 64, active_warps=8, contention_ctas=8, operation="redg_fp32", operations_per_cta_loop=8 * 9),
    ProbeSpec("barrier_balanced", 3, 5, 4096, operation="cta_barriers"),
    ProbeSpec("barrier_skew8", 3, 5, 2048, skew_iterations=8, operation="cta_barriers"),
    ProbeSpec("shared_lds128_x2", 4, 7, 2048, dynamic_smem=12_288, operation="shared_bytes", operations_per_cta_loop=THREADS * 2 * 16),
    ProbeSpec("shared_sts128_x2", 5, 7, 2048, dynamic_smem=12_288, operation="shared_bytes", operations_per_cta_loop=THREADS * 2 * 16),
    ProbeSpec("mufu_ex2_ilp4", 6, 5, 2048, operation="mufu_warp_instructions", operations_per_cta_loop=WARPS_PER_CTA * 4),
    ProbeSpec("mufu_ex2_chain", 7, 5, 4096, operation="mufu_warp_instructions", operations_per_cta_loop=WARPS_PER_CTA),
    ProbeSpec("mufu_rcp_ilp4", 8, 5, 2048, operation="mufu_warp_instructions", operations_per_cta_loop=WARPS_PER_CTA * 4),
    ProbeSpec("mufu_rcp_chain", 9, 5, 4096, operation="mufu_warp_instructions", operations_per_cta_loop=WARPS_PER_CTA),
    ProbeSpec("launch_one_cta", 10, 1, 1, dynamic_smem=0, operation="launches"),
    ProbeSpec("launch_full_grid", 10, 1, 1, dynamic_smem=0, operation="launches"),
)


NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "sm__sass_data_bytes_mem_shared_op_ld.sum",
    "sm__sass_data_bytes_mem_shared_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().split(" (", 1)[0].replace(",", "").replace("%", "")
    if not text or text in {"n/a", "N/A", "-"}:
        return math.nan
    return float(text)


NCUMetricDimension = Literal["time", "bytes"]


@dataclass(frozen=True)
class NormalizedNCUMetric:
    dimension: NCUMetricDimension
    original_value: str
    original_unit: str
    normalized_value: float
    normalized_unit: str

    def as_dict(self) -> dict[str, str | float]:
        return {
            "dimension": self.dimension,
            "original_value": self.original_value,
            "original_unit": self.original_unit,
            "normalized_value": self.normalized_value,
            "normalized_unit": self.normalized_unit,
        }


_NCU_TIME_TO_MS = {
    "ns": 1e-6,
    "nsecond": 1e-6,
    "nanosecond": 1e-6,
    "nanoseconds": 1e-6,
    "us": 1e-3,
    "usecond": 1e-3,
    "microsecond": 1e-3,
    "microseconds": 1e-3,
    "ms": 1.0,
    "msecond": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1e3,
    "second": 1e3,
    "seconds": 1e3,
}

_NCU_BYTES_TO_BYTES = {
    "B": 1.0,
    "byte": 1.0,
    "bytes": 1.0,
    "KB": 1e3,
    "kB": 1e3,
    "Kbyte": 1e3,
    "Kbytes": 1e3,
    "MB": 1e6,
    "Mbyte": 1e6,
    "Mbytes": 1e6,
    "GB": 1e9,
    "Gbyte": 1e9,
    "Gbytes": 1e9,
    "KiB": float(2**10),
    "Kibyte": float(2**10),
    "Kibytes": float(2**10),
    "MiB": float(2**20),
    "Mibyte": float(2**20),
    "Mibytes": float(2**20),
    "GiB": float(2**30),
    "Gibyte": float(2**30),
    "Gibytes": float(2**30),
}


def normalize_ncu_metric(
    value: Any, unit: str, dimension: NCUMetricDimension
) -> NormalizedNCUMetric:
    original_value = str(value)
    original_unit = str(unit).strip()
    numeric = parse_number(value)
    if dimension == "time":
        scale = _NCU_TIME_TO_MS.get(original_unit)
        if scale is None:
            raise ValueError(f"unsupported NCU time unit {original_unit!r}")
        normalized_unit = "ms"
    elif dimension == "bytes":
        base_unit, separator, qualifier = original_unit.partition("/")
        if separator and qualifier != "block":
            raise ValueError(f"unsupported NCU byte unit {original_unit!r}")
        scale = _NCU_BYTES_TO_BYTES.get(base_unit)
        if scale is None:
            raise ValueError(f"unsupported NCU byte unit {original_unit!r}")
        normalized_unit = "byte/block" if qualifier else "byte"
    else:
        raise ValueError(f"unsupported NCU metric dimension {dimension!r}")
    return NormalizedNCUMetric(
        dimension=dimension,
        original_value=original_value,
        original_unit=original_unit,
        normalized_value=numeric * scale,
        normalized_unit=normalized_unit,
    )


def _percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: Sequence[float], *, include_samples: bool = True) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("summary requires samples")
    result: dict[str, Any] = {
        "count": len(data),
        "minimum": min(data),
        "q05": _percentile(data, 5),
        "q25": _percentile(data, 25),
        "median": statistics.median(data),
        "q75": _percentile(data, 75),
        "q95": _percentile(data, 95),
        "maximum": max(data),
        "mean": statistics.fmean(data),
        "standard_deviation": statistics.pstdev(data),
    }
    if include_samples:
        result["samples"] = data
    return result


def parse_opcode_mix(value: str) -> dict[str, int]:
    match = re.search(r"\((.*)\)\s*$", value)
    if not match:
        return {}
    result: dict[str, int] = {}
    for item in match.group(1).split(";"):
        if ":" not in item:
            continue
        name, count = item.split(":", 1)
        result[name.strip()] = int(parse_number(count))
    return result


def parse_pc_instances(value: str) -> dict[int, int]:
    match = re.search(r"\((.*)\)\s*$", value)
    if not match:
        return {}
    result: dict[int, int] = {}
    for item in match.group(1).split(";"):
        if ":" not in item:
            continue
        address, count = item.strip().split(":", 1)
        if address.startswith("0x"):
            result[int(address, 16)] = int(parse_number(count))
    return result


def parse_ncu_csv(text: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    lines = [line for line in text.splitlines() if line and not line.startswith("==")]
    if not lines:
        raise ValueError("no NCU CSV rows found")
    rows = list(csv.reader(lines))
    header = rows[0]
    if "Kernel Name" not in header:
        raise ValueError("NCU CSV header is missing Kernel Name")
    units: dict[str, str] = {}
    start = 1
    if len(rows) > 1 and len(rows[1]) == len(header) and not rows[1][header.index("Kernel Name")]:
        units = dict(zip(header, rows[1]))
        start = 2
    records = [
        dict(zip(header, row + [""] * (len(header) - len(row))))
        for row in rows[start:]
        if len(row) <= len(header)
    ]
    return records, units


SASS_FUNCTION_RE = re.compile(r"Function\s*:\s*(\S+)")
SASS_INSTRUCTION_RE = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(.*)")
SASS_OPCODE_RE = re.compile(
    r"\b(BAR(?:\.[A-Z0-9_]+)*|LDS(?:\.[A-Z0-9_]+)*|STS(?:\.[A-Z0-9_]+)*|"
    r"SHFL(?:\.[A-Z0-9_]+)*|REDG(?:\.[A-Z0-9_]+)*|REDUX(?:\.[A-Z0-9_]+)*|MUFU(?:\.[A-Z0-9_]+)*|"
    r"FADD(?:\.[A-Z0-9_]+)*|FFMA(?:\.[A-Z0-9_]+)*|FMUL(?:\.[A-Z0-9_]+)*)\b"
)


def parse_sass_functions(text: str) -> dict[str, list[dict[str, Any]]]:
    functions: dict[str, list[dict[str, Any]]] = {}
    current: str | None = None
    for line in text.splitlines():
        function_match = SASS_FUNCTION_RE.search(line)
        if function_match:
            current = function_match.group(1)
            functions.setdefault(current, [])
            continue
        if current is None:
            continue
        instruction_match = SASS_INSTRUCTION_RE.search(line)
        if not instruction_match:
            continue
        body = instruction_match.group(2)
        opcode_match = SASS_OPCODE_RE.search(body)
        opcode = opcode_match.group(1) if opcode_match else "OTHER"
        functions[current].append(
            {
                "offset": int(instruction_match.group(1), 16),
                "opcode": opcode,
                "text": body.strip(),
            }
        )
    return functions


def select_sass_function(
    functions: dict[str, list[dict[str, Any]]], substring: str
) -> tuple[str, list[dict[str, Any]]]:
    matches = [(name, rows) for name, rows in functions.items() if substring in name]
    if len(matches) != 1:
        raise ValueError(f"expected one SASS function containing {substring!r}, got {len(matches)}")
    return matches[0]


def dynamic_subopcode_counts(
    pc_counts: dict[int, int], instructions: Sequence[dict[str, Any]]
) -> dict[str, int]:
    if not pc_counts or not instructions:
        return {}
    base = min(pc_counts) - min(int(row["offset"]) for row in instructions)
    by_offset = {int(row["offset"]): str(row["opcode"]) for row in instructions}
    result: dict[str, int] = {}
    for pc, count in pc_counts.items():
        opcode = by_offset.get(pc - base)
        if opcode is not None:
            result[opcode] = result.get(opcode, 0) + count
    return result


def reduction_algorithmic_counts(
    *, dynamic_shuffles: int, launched_warps: int, reductions_per_event: int = 9
) -> dict[str, Any]:
    event_shuffles = dynamic_shuffles
    divisor = reductions_per_event * 5
    if event_shuffles < 0 or event_shuffles % divisor:
        raise ValueError(
            f"SHFL count {dynamic_shuffles} is inconsistent with the exact reduction structure"
        )
    events = event_shuffles // divisor
    return {
        "launched_warps": launched_warps,
        "initial_warp_max_reductions": launched_warps,
        "initial_warp_max_redux_instructions": launched_warps,
        "warp_active_gaussian_events": events,
        "sum_reductions_per_event": reductions_per_event,
        "shuffle_add_pairs_per_event": divisor,
        "sum_shuffle_add_pairs": event_shuffles,
        "total_shuffle_add_pairs": dynamic_shuffles,
        "minimum_redg_fp32_atomics": events * reductions_per_event,
    }


def independent_resource_bound_ms(resources_ms: dict[str, float]) -> dict[str, Any]:
    finite = {name: value for name, value in resources_ms.items() if math.isfinite(value)}
    if not finite:
        raise ValueError("resource model has no finite terms")
    limiting = max(finite, key=finite.get)
    return {
        "equation": "max(" + ", ".join(finite) + ")",
        "terms_ms": finite,
        "limiting_resource": limiting,
        "lower_bound_ms": finite[limiting],
    }


def criteria_from_ranges(
    *, resource_target_low_ms: float, current_q95_ms: float, current_q05_ms: float,
    resource_target_high_ms: float
) -> dict[str, Any]:
    robust_efficiency = 100.0 * resource_target_low_ms / current_q95_ms
    worst_residual = 100.0 * (current_q95_ms / resource_target_low_ms - 1.0)
    best_residual = max(
        0.0, 100.0 * (current_q05_ms / resource_target_high_ms - 1.0)
    )
    return {
        "robust_minimum_efficiency_percent": robust_efficiency,
        "residual_relative_to_resource_target_percent_range": [
            best_residual,
            worst_residual,
        ],
        "at_least_25_percent_of_ceiling_established": robust_efficiency >= 25.0,
        "less_than_10_percent_residual_established": worst_residual < 10.0,
    }


def _load_extension() -> Any:
    import torch
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="ptxsplat_kernel_ceiling_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        with_cuda=True,
        verbose=False,
    )


def _telemetry() -> dict[str, Any]:
    fields = (
        "timestamp,name,driver_version,temperature.gpu,clocks.current.sm,"
        "clocks.current.memory,power.draw,utilization.gpu"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(command, text=True).strip().split(", ")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False}
    names = fields.split(",")
    result: dict[str, Any] = {"available": True}
    for name, value in zip(names, values):
        try:
            result[name] = float(value)
        except ValueError:
            result[name] = value
    return result


def _event_samples(
    operation: Callable[[], Any], *, warmup: int, samples: int, repetitions: int
) -> list[float]:
    import torch

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repetitions)
    return values


def _warm_device(seconds: float) -> None:
    import torch

    if seconds <= 0:
        return
    left = torch.randn((4096, 4096), device="cuda")
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        torch.mm(left, right, out=output)
        torch.cuda.synchronize()


def _probe_output_elements(spec: ProbeSpec, sm_count: int) -> int:
    if spec.kind == 2:
        return math.ceil(spec.blocks(sm_count) / spec.contention_ctas) * 9
    return max(spec.blocks(sm_count) * THREADS, 1)


def _run_probe(extension: Any, output: Any, spec: ProbeSpec, sm_count: int) -> None:
    extension.run_probe(
        output,
        spec.kind,
        spec.blocks(sm_count),
        spec.loops,
        spec.active_warps,
        spec.contention_ctas,
        spec.skew_iterations,
        spec.dynamic_smem,
    )


def _measure_probes(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    import torch

    extension = _load_extension()
    _warm_device(args.thermal_warmup_seconds)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    outputs = {
        spec.name: torch.zeros(
            _probe_output_elements(spec, sm_count), device="cuda", dtype=torch.float32
        )
        for spec in PROBE_SPECS
    }
    samples: dict[str, list[float]] = {spec.name: [] for spec in PROBE_SPECS}
    samples_by_round: dict[str, list[list[float]]] = {
        spec.name: [] for spec in PROBE_SPECS
    }
    telemetry: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for round_index in range(args.rounds):
        order = list(PROBE_SPECS)
        rng.shuffle(order)
        round_before = _telemetry()
        for spec in order:
            operation = lambda spec=spec: _run_probe(
                extension, outputs[spec.name], spec, sm_count
            )
            round_values = _event_samples(
                operation,
                warmup=args.probe_warmup,
                samples=args.samples_per_round,
                repetitions=args.repetitions_per_sample,
            )
            samples[spec.name].extend(round_values)
            samples_by_round[spec.name].append(round_values)
        telemetry.append(
            {
                "round": round_index,
                "probe_order": [spec.name for spec in order],
                "before": round_before,
                "after": _telemetry(),
            }
        )
    available_start_clocks = [
        float(item["before"]["clocks.current.sm"])
        for item in telemetry
        if item["before"].get("available")
        and isinstance(item["before"].get("clocks.current.sm"), (int, float))
    ]
    maximum_start_clock = max(available_start_clocks, default=0.0)
    accepted_rounds = [
        item["round"]
        for item in telemetry
        if (
            not available_start_clocks
            or (
                float(item["before"].get("clocks.current.sm", 0.0))
                >= 0.90 * maximum_start_clock
                and float(item["before"].get("temperature.gpu", 0.0)) <= 85.0
            )
        )
    ]
    if not accepted_rounds:
        raise RuntimeError("no thermally stable microbenchmark rounds were accepted")
    probes: dict[str, Any] = {}
    for spec in PROBE_SPECS:
        timing = summarize(samples[spec.name])
        selected_samples = [
            value
            for round_index in accepted_rounds
            for value in samples_by_round[spec.name][round_index]
        ]
        selected_timing = summarize(selected_samples)
        operations = spec.operations(sm_count)
        rates = [operations / (sample * 1e-3) for sample in samples[spec.name]]
        selected_rates = [
            operations / (sample * 1e-3) for sample in selected_samples
        ]
        probes[spec.name] = {
            "configuration": {
                "kernel_kind": spec.kind,
                "blocks": spec.blocks(sm_count),
                "threads_per_block": THREADS,
                "target_ctas_per_sm": spec.blocks_per_sm,
                "loops": spec.loops,
                "active_warps": spec.active_warps,
                "contention_ctas_per_address_group": spec.contention_ctas,
                "skew_dependency_iterations": spec.skew_iterations,
                "dynamic_shared_memory_bytes": spec.dynamic_smem,
            },
            "counting": {
                "operation": spec.operation,
                "operations_per_launch": operations,
            },
            "timing_ms": timing,
            "sustained_rate_per_second": summarize(rates),
            "round_timing_ms": [
                summarize(round_values)
                for round_values in samples_by_round[spec.name]
            ],
            "thermal_controlled_timing_ms": selected_timing,
            "thermal_controlled_sustained_rate_per_second": summarize(
                selected_rates
            ),
            "checksum": float(outputs[spec.name].sum().cpu()),
        }
    for spec in (item for item in PROBE_SPECS if item.kind == 10):
        host_samples: list[float] = []
        for _ in range(30):
            torch.cuda.synchronize()
            start_ns = time.perf_counter_ns()
            for _ in range(100):
                _run_probe(extension, outputs[spec.name], spec, sm_count)
            elapsed_ns = time.perf_counter_ns() - start_ns
            torch.cuda.synchronize()
            host_samples.append(elapsed_ns / 100 / 1e3)
        probes[spec.name]["host_enqueue_us"] = summarize(host_samples)
    return {
        "thermal_policy": {
            "method": "fixed GEMM warmup, randomized probe order per round, telemetry bracketing each round",
            "warmup_seconds": args.thermal_warmup_seconds,
            "clock_locked": False,
            "reason_clock_not_locked": "benchmark container does not assume host administrator privileges",
            "round_acceptance_rule": "starting SM clock >= 90% of the maximum round-start clock and temperature <= 85 C",
            "maximum_round_start_sm_clock_mhz": maximum_start_clock,
            "accepted_rounds": accepted_rounds,
            "rejected_rounds_retained_in_raw_samples": [
                item["round"]
                for item in telemetry
                if item["round"] not in accepted_rounds
            ],
            "telemetry": telemetry,
        },
        "probes": probes,
    }, str(Path(extension.__file__).resolve())


def _build_raster_workload() -> dict[str, Any]:
    import torch

    from benchmarks.garden import _colors_for_mode
    from ptxsplat._helper import load_test_data
    from ptxsplat.cuda._backend import _C
    from ptxsplat.rendering import rasterization

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    (
        means,
        quats,
        scales,
        opacities,
        rgb,
        viewmats,
        intrinsics,
        fixture_width,
        fixture_height,
    ) = load_test_data(
        data_path="assets/test_garden.npz", device="cuda", scene_grid=7
    )
    viewmats = viewmats[:1].contiguous()
    intrinsics = intrinsics[:1].clone()
    intrinsics[..., 0, :] *= WIDTH / fixture_width
    intrinsics[..., 1, :] *= HEIGHT / fixture_height
    sh_colors, sh_degree = _colors_for_mode(rgb, "sh3")
    background = torch.zeros((1, 3), device="cuda")
    _, _, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        sh_colors,
        viewmats,
        intrinsics,
        WIDTH,
        HEIGHT,
        sh_degree=sh_degree,
        packed=True,
        backgrounds=background,
        render_mode="RGB",
        camera_model="pinhole",
    )
    raster_colors = rgb[meta["gaussian_ids"]].contiguous()

    def forward() -> tuple[Any, Any, Any]:
        return _C.rasterize_to_pixels_3dgs_fwd(
            meta["means2d"],
            meta["conics"],
            raster_colors,
            meta["opacities"],
            background,
            None,
            WIDTH,
            HEIGHT,
            TILE_SIZE,
            meta["isect_offsets"],
            meta["flatten_ids"],
        )

    renders, alphas, last_ids = forward()
    v_renders = 2.0 * (renders - 0.5) / renders.numel()
    v_alphas = torch.zeros_like(alphas)

    def backward() -> tuple[Any, Any, Any, Any, Any]:
        return _C.rasterize_to_pixels_3dgs_bwd(
            meta["means2d"],
            meta["conics"],
            raster_colors,
            meta["opacities"],
            background,
            None,
            WIDTH,
            HEIGHT,
            TILE_SIZE,
            meta["isect_offsets"],
            meta["flatten_ids"],
            alphas,
            last_ids,
            v_renders,
            v_alphas,
            False,
        )

    return {
        "forward": forward,
        "backward": backward,
        "case": {
            "seed": SEED,
            "scene_grid": 7,
            "resolution": [WIDTH, HEIGHT],
            "tile_size": TILE_SIZE,
            "grid_blocks": GRID_BLOCKS,
            "color_mode": "sh3",
            "background": "black",
            "packed": True,
            "gaussian_count": int(means.shape[0]),
            "visible_gaussian_count": int(meta["radii"].numel()),
            "intersection_count": int(meta["flatten_ids"].numel()),
            "nonempty_tiles": int(
                (torch.diff(
                    torch.cat(
                        [
                            meta["isect_offsets"].reshape(-1),
                            torch.tensor(
                                [meta["flatten_ids"].numel()],
                                device="cuda",
                                dtype=meta["isect_offsets"].dtype,
                            ),
                        ]
                    )
                ) > 0).sum().cpu()
            ),
            "forward_checksums": {
                "render": float(renders.sum().cpu()),
                "alpha": float(alphas.sum().cpu()),
                "last_id": int(last_ids.to(torch.int64).sum().cpu()),
            },
        },
    }


def _measure_raster(args: argparse.Namespace) -> dict[str, Any]:
    workload = _build_raster_workload()
    result: dict[str, Any] = {"case": workload["case"], "telemetry": []}
    for name in ("forward", "backward"):
        before = _telemetry()
        samples = _event_samples(
            workload[name],
            warmup=args.raster_warmup,
            samples=args.raster_samples,
            repetitions=1,
        )
        result[name] = {
            "scope": (
                "public native wrapper; forward contains one promoted kernel launch, "
                "backward additionally contains output zero-initialization"
            ),
            "timing_ms": summarize(samples),
        }
        result["telemetry"].append(
            {"stage": name, "before": before, "after": _telemetry()}
        )
    return result


def _environment() -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_porcelain": _git(["status", "--porcelain"]),
        "container_image": os.environ.get("PTXSPLAT_DOCKER_IMAGE"),
        "container_image_id": os.environ.get("PTXSPLAT_DOCKER_IMAGE_ID"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "l2_cache_bytes": properties.L2_cache_size,
        "telemetry": _telemetry(),
    }


def command_measure(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    microbenchmarks, extension_path = _measure_probes(args)
    raster = _measure_raster(args)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "kernel-ceiling-measurements",
        "generated_at_utc": _utc_now(),
        "arguments": {
            key: value for key, value in vars(args).items() if key != "function"
        },
        "environment": _environment(),
        "extension_path": extension_path,
        "microbenchmarks": microbenchmarks,
        "raster": raster,
    }
    path = output_dir / "measurements.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


def command_profile_target(args: argparse.Namespace) -> int:
    import torch

    if args.target == "micro":
        extension = _load_extension()
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        outputs = {
            spec.name: torch.zeros(
                _probe_output_elements(spec, sm_count), device="cuda", dtype=torch.float32
            )
            for spec in PROBE_SPECS
        }
        for spec in PROBE_SPECS:
            _run_probe(extension, outputs[spec.name], spec, sm_count)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for spec in PROBE_SPECS:
            _run_probe(extension, outputs[spec.name], spec, sm_count)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(json.dumps({"target": args.target, "telemetry": _telemetry()}))
        return 0

    workload = _build_raster_workload()
    operation = workload[args.target]
    for _ in range(args.target_warmup):
        operation()
    torch.cuda.synchronize()
    before = _telemetry()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.target_launches):
        operation()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        json.dumps(
            {
                "target": args.target,
                "launches": args.target_launches,
                "case": workload["case"],
                "telemetry_before": before,
                "telemetry_after": _telemetry(),
            },
            sort_keys=True,
        )
    )
    return 0


def _capture_selected_sass(binary: Path, output: Path, substrings: Sequence[str]) -> None:
    process = subprocess.Popen(
        ["cuobjdump", "--dump-sass", str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    selected: list[str] = []
    keep = False
    for line in process.stdout:
        match = SASS_FUNCTION_RE.search(line)
        if match:
            keep = any(substring in match.group(1) for substring in substrings)
        if keep:
            selected.append(line)
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"cuobjdump failed ({return_code}): {stderr}")
    output.write_text("".join(selected))


def _run_logged(
    command: Sequence[str], *, output: Path, env: dict[str, str] | None = None
) -> None:
    with output.open("w") as stream:
        completed = subprocess.run(
            list(command), stdout=stream, stderr=subprocess.STDOUT, text=True, env=env
        )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def _ncu_profile_command(
    *, target: str, report: Path, launches: int
) -> list[str]:
    kernel_filter = {
        "forward": "regex:rasterize_to_pixels_3dgs_fwd_sm120",
        "backward": "regex:rasterize_to_pixels_3dgs_bwd_sm120",
        "micro": "regex:kc_",
    }[target]
    count = launches if target != "micro" else len(PROBE_SPECS)
    return [
        "ncu",
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--section",
        "InstructionStats",
        "--metrics",
        ",".join(NCU_METRICS),
        "--kernel-name-base",
        "function",
        "--kernel-name",
        kernel_filter,
        "--launch-count",
        str(count),
        "--force-overwrite",
        "--export",
        str(report),
        sys.executable,
        "-m",
        "benchmarks.kernel_ceiling",
        "profile-target",
        "--target",
        target,
        "--target-launches",
        str(launches),
    ]


def _ncu_export(report: Path, output: Path, page: str) -> list[str]:
    command = [
        "ncu",
        "--import",
        str(report),
        "--csv",
        "--print-units",
        "base",
        "--print-fp",
        "--print-metric-instances",
        "details",
    ]
    if page == "details":
        command.extend(["--print-details", "all"])
    command.extend(["--page", page])
    _run_logged(command, output=output)
    return command


def command_run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    measure_command = [
        sys.executable,
        "-m",
        "benchmarks.kernel_ceiling",
        "measure",
        "--output-dir",
        str(output_dir),
        "--thermal-warmup-seconds",
        str(args.thermal_warmup_seconds),
        "--rounds",
        str(args.rounds),
        "--samples-per-round",
        str(args.samples_per_round),
        "--repetitions-per-sample",
        str(args.repetitions_per_sample),
        "--raster-samples",
        str(args.raster_samples),
    ]
    commands.append(measure_command)
    _run_logged(measure_command, output=output_dir / "measure.log", env=os.environ.copy())
    measurements = json.loads((output_dir / "measurements.json").read_text())

    ptx_binary = Path("ptxsplat/csrc.so").resolve()
    micro_binary = Path(measurements["extension_path"])
    _capture_selected_sass(
        ptx_binary,
        output_dir / "raster.sass",
        ("rasterize_to_pixels_3dgs_fwd_sm120", "rasterize_to_pixels_3dgs_bwd_sm120"),
    )
    _capture_selected_sass(
        micro_binary,
        output_dir / "microbench.sass",
        tuple(f"kc_{name}" for name in (
            "reduction", "redux", "atomic", "barrier", "shared", "mufu", "empty"
        )),
    )
    commands.extend(
        [
            ["cuobjdump", "--dump-sass", str(ptx_binary), "|", "target-function-filter"],
            ["cuobjdump", "--dump-sass", str(micro_binary), "|", "target-function-filter"],
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")
    env["PTXSPLAT_BACKEND"] = "sm120"
    for target in ("forward", "backward", "micro"):
        report = output_dir / f"ncu-{target}"
        command = _ncu_profile_command(
            target=target, report=report, launches=args.ncu_raster_launches
        )
        commands.append(command)
        _run_logged(command, output=output_dir / f"ncu-{target}.log", env=env)
        report_file = report.with_suffix(".ncu-rep")
        commands.append(_ncu_export(report_file, output_dir / f"ncu-{target}-raw.csv", "raw"))
        commands.append(
            _ncu_export(report_file, output_dir / f"ncu-{target}-details.csv", "details")
        )

    command_file = output_dir / "commands.json"
    command_file.write_text(json.dumps(commands, indent=2) + "\n")
    analysis_args = argparse.Namespace(output_dir=str(output_dir))
    command_analyze(analysis_args)
    return 0


def _ncu_kernel_summary(
    raw_text: str,
    details_text: str,
    sass_instructions: Sequence[dict[str, Any]],
    kernel_substring: str,
) -> dict[str, Any]:
    raw_rows, units = parse_ncu_csv(raw_text)
    rows = [row for row in raw_rows if kernel_substring in row.get("Kernel Name", "")]
    if not rows:
        raise ValueError(f"NCU raw report has no kernel containing {kernel_substring!r}")
    details_rows, _ = parse_ncu_csv(details_text)
    detail_rows = [
        row for row in details_rows if kernel_substring in row.get("Kernel Name", "")
    ]
    opcode_mixes = [
        parse_opcode_mix(row["Metric Value"])
        for row in detail_rows
        if row.get("Metric Name") == "Executed Warp-Level Instructions By Basic SASS Opcode"
    ]
    pc_instances = [
        parse_pc_instances(row["Metric Value"])
        for row in detail_rows
        if row.get("Metric Name") == "Instructions Executed" and "0x" in row.get("Metric Value", "")
    ]
    opcode_mix = opcode_mixes[0] if opcode_mixes else {}
    subopcodes = (
        dynamic_subopcode_counts(pc_instances[0], sass_instructions)
        if pc_instances
        else {}
    )

    def values(name: str) -> list[float]:
        return [parse_number(row[name]) for row in rows if row.get(name, "")]

    durations = [
        normalize_ncu_metric(
            row["gpu__time_duration.sum"], units["gpu__time_duration.sum"], "time"
        )
        for row in rows
    ]
    dram_bytes = [
        normalize_ncu_metric(
            row["dram__bytes.sum"], units["dram__bytes.sum"], "bytes"
        )
        for row in rows
    ]
    l2_bytes = [
        normalize_ncu_metric(
            row["lts__t_bytes.sum"], units["lts__t_bytes.sum"], "bytes"
        )
        for row in rows
    ]
    shared_load_bytes = [
        normalize_ncu_metric(
            row["sm__sass_data_bytes_mem_shared_op_ld.sum"],
            units["sm__sass_data_bytes_mem_shared_op_ld.sum"],
            "bytes",
        )
        for row in rows
    ]
    shared_store_bytes = [
        normalize_ncu_metric(
            row["sm__sass_data_bytes_mem_shared_op_st.sum"],
            units["sm__sass_data_bytes_mem_shared_op_st.sum"],
            "bytes",
        )
        for row in rows
    ]
    result = {
        "kernel": rows[0]["Kernel Name"],
        "launches_profiled": len(rows),
        "duration_ms": summarize(
            [metric.normalized_value for metric in durations]
        ),
        "dram_bytes": statistics.median(
            metric.normalized_value for metric in dram_bytes
        ),
        "l2_bytes": statistics.median(
            metric.normalized_value for metric in l2_bytes
        ),
        "unit_normalization_provenance": {
            "duration": [metric.as_dict() for metric in durations],
            "dram": [metric.as_dict() for metric in dram_bytes],
            "l2": [metric.as_dict() for metric in l2_bytes],
            "shared_load": [metric.as_dict() for metric in shared_load_bytes],
            "shared_store": [metric.as_dict() for metric in shared_store_bytes],
        },
        "l2_throughput_pct_of_peak": statistics.median(
            values("lts__throughput.avg.pct_of_peak_sustained_elapsed")
        ),
        "achieved_occupancy_pct": statistics.median(
            values("sm__warps_active.avg.pct_of_peak_sustained_active")
        ),
        "fp32_thread_instructions": {
            "fadd": statistics.median(values("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")),
            "ffma": statistics.median(values("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")),
            "fmul": statistics.median(values("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")),
            "all_fp32": statistics.median(values("smsp__sass_thread_inst_executed_op_fp32_pred_on.sum")),
        },
        "shared_data_bytes": {
            "load": statistics.median(
                metric.normalized_value for metric in shared_load_bytes
            ),
            "store": statistics.median(
                metric.normalized_value for metric in shared_store_bytes
            ),
        },
        "global_reduction_requests": statistics.median(
            values("l1tex__t_requests_pipe_lsu_mem_global_op_red.sum")
        ),
        "dynamic_basic_opcode_counts": opcode_mix,
        "dynamic_sass_subopcode_counts": subopcodes,
    }
    fp = result["fp32_thread_instructions"]
    result["counted_flops_fadd_plus_fmul_plus_2xffma"] = (
        fp["fadd"] + fp["fmul"] + 2.0 * fp["ffma"]
    )
    return result


MICRO_KERNEL_BY_KIND = {
    0: "kc_reduction_ilp9",
    1: "kc_reduction_chain",
    2: "kc_atomic_redg",
    3: "kc_barrier",
    4: "kc_shared_lds128",
    5: "kc_shared_sts128",
    6: "kc_mufu_ex2_ilp4",
    7: "kc_mufu_ex2_chain",
    8: "kc_mufu_rcp_ilp4",
    9: "kc_mufu_rcp_chain",
    10: "kc_empty_grid",
    11: "kc_redux_max_chain",
}


def _micro_dynamic_expectation(spec: ProbeSpec, sm_count: int) -> dict[str, Any]:
    blocks = spec.blocks(sm_count)
    if spec.kind in (0, 1):
        count = spec.operations(sm_count) * (5 if spec.kind == 1 else 1)
        return {"opcode": "SHFL", "minimum_count": count}
    if spec.kind == 11:
        return {"opcode": "REDUX", "minimum_count": spec.operations(sm_count)}
    if spec.kind == 2:
        return {"opcode": "REDG", "minimum_count": spec.operations(sm_count)}
    if spec.kind == 3:
        return {
            "opcode": "BAR",
            "minimum_count": blocks * spec.loops * WARPS_PER_CTA,
        }
    if spec.kind == 4:
        return {
            "opcode": "LDS",
            "minimum_count": blocks * spec.loops * WARPS_PER_CTA * 2,
        }
    if spec.kind == 5:
        return {
            "opcode": "STS",
            "minimum_count": blocks * spec.loops * WARPS_PER_CTA * 2,
        }
    if spec.kind in (6, 7, 8, 9):
        return {"opcode": "MUFU", "minimum_count": spec.operations(sm_count)}
    return {"opcode": None, "minimum_count": 0}


def _ncu_micro_summaries(
    raw_text: str,
    details_text: str,
    micro_functions: dict[str, list[dict[str, Any]]],
    sm_count: int,
) -> dict[str, Any]:
    raw_rows, units = parse_ncu_csv(raw_text)
    rows = [row for row in raw_rows if "kc_" in row.get("Kernel Name", "")]
    if len(rows) != len(PROBE_SPECS):
        raise ValueError(
            f"expected {len(PROBE_SPECS)} profiled micro kernels, got {len(rows)}"
        )
    rows.sort(key=lambda row: int(row["ID"]))
    detail_rows, _ = parse_ncu_csv(details_text)
    result: dict[str, Any] = {}
    for spec, row in zip(PROBE_SPECS, rows):
        expected_kernel = MICRO_KERNEL_BY_KIND[spec.kind]
        if expected_kernel not in row["Kernel Name"]:
            raise ValueError(
                f"micro launch order mismatch for {spec.name}: {row['Kernel Name']}"
            )
        launch_details = [
            item
            for item in detail_rows
            if item.get("ID") == row["ID"] and expected_kernel in item.get("Kernel Name", "")
        ]
        opcode_row = next(
            (
                item
                for item in launch_details
                if item.get("Metric Name")
                == "Executed Warp-Level Instructions By Basic SASS Opcode"
            ),
            None,
        )
        pc_row = next(
            (
                item
                for item in launch_details
                if item.get("Metric Name") == "Instructions Executed"
                and "0x" in item.get("Metric Value", "")
            ),
            None,
        )
        opcode_mix = parse_opcode_mix(opcode_row["Metric Value"]) if opcode_row else {}
        function_name, instructions = select_sass_function(
            micro_functions, expected_kernel
        )
        subopcodes = (
            dynamic_subopcode_counts(
                parse_pc_instances(pc_row["Metric Value"]), instructions
            )
            if pc_row
            else {}
        )
        expected = _micro_dynamic_expectation(spec, sm_count)
        observed = (
            opcode_mix.get(expected["opcode"], 0)
            if expected["opcode"] is not None
            else 0
        )
        duration = normalize_ncu_metric(
            row["gpu__time_duration.sum"], units["gpu__time_duration.sum"], "time"
        )
        dynamic_shared_memory = normalize_ncu_metric(
            row["launch__shared_mem_per_block_dynamic"],
            units["launch__shared_mem_per_block_dynamic"],
            "bytes",
        )
        result[spec.name] = {
            "kernel": row["Kernel Name"],
            "sass_function": function_name,
            "duration_ms": duration.normalized_value,
            "achieved_occupancy_pct": parse_number(
                row["sm__warps_active.avg.pct_of_peak_sustained_active"]
            ),
            "launch_resources": {
                "registers_per_thread": parse_number(
                    row["launch__registers_per_thread"]
                ),
                "allocated_registers_per_thread": parse_number(
                    row["launch__registers_per_thread_allocated"]
                ),
                "dynamic_shared_memory_bytes": dynamic_shared_memory.normalized_value,
                "occupancy_limit_register_ctas": parse_number(
                    row["launch__occupancy_limit_registers"]
                ),
                "occupancy_limit_shared_memory_ctas": parse_number(
                    row["launch__occupancy_limit_shared_mem"]
                ),
                "occupancy_limit_warp_ctas": parse_number(
                    row["launch__occupancy_limit_warps"]
                ),
            },
            "unit_normalization_provenance": {
                "duration": duration.as_dict(),
                "dynamic_shared_memory": dynamic_shared_memory.as_dict(),
            },
            "dynamic_basic_opcode_counts": opcode_mix,
            "dynamic_sass_subopcode_counts": subopcodes,
            "expected_dynamic_execution": {
                **expected,
                "observed_count": observed,
                "observed_at_least_minimum": observed >= expected["minimum_count"],
            },
        }
    return result


def _sass_proof(
    instructions: Sequence[dict[str, Any]], expected: dict[str, int]
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in instructions:
        opcode = str(row["opcode"])
        counts[opcode] = counts.get(opcode, 0) + 1
    checks = {
        prefix: sum(count for opcode, count in counts.items() if opcode.startswith(prefix)) >= minimum
        for prefix, minimum in expected.items()
    }
    snippets = [
        {"offset_hex": hex(int(row["offset"])), "opcode": row["opcode"], "text": row["text"]}
        for row in instructions
        if any(str(row["opcode"]).startswith(prefix) for prefix in expected)
    ]
    return {
        "static_opcode_counts": counts,
        "required_minimums": expected,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "instruction_snippets": snippets,
    }


def _rate(probes: dict[str, Any], name: str, quantile: str = "median") -> float:
    return float(
        probes[name]["thermal_controlled_sustained_rate_per_second"][quantile]
    )


def _kernel_model(
    ncu: dict[str, Any], probes: dict[str, Any], *, stage: str,
    fp32_ceiling_tflops: float, dram_ceiling_gbs: float,
    rate_quantile: str, contention_probe: str, barrier_probe: str,
    chain_mufu: bool, launch_latency_quantile: str,
    current_wrapper_duration_ms: float,
) -> dict[str, Any]:
    profiled_duration_ms = ncu["duration_ms"]["median"]
    l2_achieved_gbs = ncu["l2_bytes"] / (profiled_duration_ms * 1e-3) / 1e9
    l2_ceiling_gbs = l2_achieved_gbs / (ncu["l2_throughput_pct_of_peak"] / 100.0)
    old_terms = {
        "fp32": ncu["counted_flops_fadd_plus_fmul_plus_2xffma"] / (fp32_ceiling_tflops * 1e12) * 1e3,
        "dram": ncu["dram_bytes"] / (dram_ceiling_gbs * 1e9) * 1e3,
        "l2": ncu["l2_bytes"] / (l2_ceiling_gbs * 1e9) * 1e3,
    }
    old = independent_resource_bound_ms(old_terms)
    dynamic = ncu["dynamic_sass_subopcode_counts"]
    basic = ncu["dynamic_basic_opcode_counts"]
    shared = ncu["shared_data_bytes"]
    lds_rate = _rate(probes, "shared_lds128_x2", rate_quantile)
    sts_rate = _rate(probes, "shared_sts128_x2", rate_quantile)
    # Loads and stores use the same shared-memory path and are non-overlapping demands.
    shared_ms = (shared["load"] / lds_rate + shared["store"] / sts_rate) * 1e3
    barrier_warp_instructions = sum(
        count for opcode, count in dynamic.items() if opcode.startswith("BAR")
    )
    barrier_ctas = barrier_warp_instructions / WARPS_PER_CTA
    barrier_ms = barrier_ctas / _rate(probes, barrier_probe, rate_quantile) * 1e3
    ex2_count = sum(
        count for opcode, count in dynamic.items() if opcode.startswith("MUFU.EX2")
    )
    rcp_count = sum(
        count for opcode, count in dynamic.items() if opcode.startswith("MUFU.RCP")
    )
    ex2_probe = "mufu_ex2_chain" if chain_mufu else "mufu_ex2_ilp4"
    rcp_probe = "mufu_rcp_chain" if chain_mufu else "mufu_rcp_ilp4"
    # EX2 and RCP share the MUFU issue resource, so their exclusive demands add.
    mufu_ms = (
        ex2_count / _rate(probes, ex2_probe, rate_quantile)
        + rcp_count / _rate(probes, rcp_probe, rate_quantile)
    ) * 1e3
    launch_ms = probes["launch_full_grid"]["thermal_controlled_timing_ms"][
        launch_latency_quantile
    ]
    operation_terms = {
        **old_terms,
        "shared_lds_sts": shared_ms,
        "cta_barrier": barrier_ms,
        "mufu_ex2_rcp": mufu_ms,
        "full_grid_launch_and_tail": launch_ms,
    }
    algorithmic_counts: dict[str, Any] = {
        "dynamic_barrier_warp_instructions": barrier_warp_instructions,
        "dynamic_barrier_subopcode_warp_instructions": {
            opcode: count
            for opcode, count in dynamic.items()
            if opcode.startswith("BAR")
        },
        "minimum_cta_barriers": barrier_ctas,
        "minimum_shared_load_bytes": shared["load"],
        "minimum_shared_store_bytes": shared["store"],
        "minimum_mufu_ex2_warp_instructions": ex2_count,
        "minimum_mufu_rcp_warp_instructions": rcp_count,
    }
    if stage == "backward":
        shuffle_count = int(basic.get("SHFL", 0))
        reductions = reduction_algorithmic_counts(
            dynamic_shuffles=shuffle_count,
            launched_warps=GRID_BLOCKS * WARPS_PER_CTA,
        )
        atomic_count = int(basic.get("REDG", 0))
        redux_count = int(basic.get("REDUX", 0))
        reductions["dynamic_redg_fp32_atomics"] = atomic_count
        reductions["dynamic_warp_max_redux_instructions"] = redux_count
        reductions["atomic_count_matches_exact_minimum"] = (
            atomic_count == reductions["minimum_redg_fp32_atomics"]
        )
        reductions["warp_max_count_matches_exact_minimum"] = (
            redux_count == reductions["initial_warp_max_redux_instructions"]
        )
        reduction_issue_ms = (
            reductions["total_shuffle_add_pairs"]
            / _rate(probes, "reduction_ilp9", rate_quantile)
            * 1e3
        )
        reduction_depth_ms = (
            reductions["warp_active_gaussian_events"]
            / _rate(probes, "reduction_chain", rate_quantile)
            * 1e3
        )
        redux_dependency_ms = (
            redux_count / _rate(probes, "redux_max_chain", rate_quantile) * 1e3
        )
        atomic_ms = atomic_count / _rate(probes, contention_probe, rate_quantile) * 1e3
        operation_terms.update(
            {
                "warp_reduction_issue": reduction_issue_ms,
                "warp_reduction_five_stage_dependency": reduction_depth_ms,
                "warp_redux_max_dependency": redux_dependency_ms,
                "contended_redg_fp32_atomic": atomic_ms,
            }
        )
        algorithmic_counts["warp_reduction_and_atomic"] = reductions
    throughput_floor_applied: dict[str, float] = {}
    for resource in tuple(operation_terms):
        if (
            resource not in old_terms
            and operation_terms[resource] > current_wrapper_duration_ms
        ):
            throughput_floor_applied[resource] = operation_terms[resource]
            operation_terms[resource] = current_wrapper_duration_ms
    operation_bound = independent_resource_bound_ms(operation_terms)
    operation_target = {
        "equation": operation_bound["equation"],
        "terms_ms": operation_bound["terms_ms"],
        "limiting_resource": operation_bound["limiting_resource"],
        "target_ms": operation_bound["lower_bound_ms"],
        "interpretation": (
            "Empirical sustained resource target from independent probe rates. "
            "It is not a physical latency lower bound or an achievable direct-PTX claim."
        ),
    }
    return {
        "current_non_profiled_wrapper_duration_ms": current_wrapper_duration_ms,
        "ncu_profiled_duration_ms": profiled_duration_ms,
        "ncu_derived_l2_ceiling_gb_s": l2_ceiling_gbs,
        "optimistic_l2_only_roofline": old,
        "operation_aware_empirical_sustained_resource_target": operation_target,
        "observed_exact_kernel_throughput_floor": {
            "rule": "A ceiling rate cannot be lower than the rate already sustained by the exact kernel; candidate terms above the non-profiled wrapper duration are capped at that conservative upper bound.",
            "uncapped_terms_ms": throughput_floor_applied,
        },
        "algorithmic_minimum_counts": algorithmic_counts,
    }


def _combine_stage_models(
    models: dict[str, dict[str, Any]], key: str, value_key: str
) -> dict[str, Any]:
    forward = models["forward"][key][value_key]
    backward = models["backward"][key][value_key]
    return {
        "equation": f"forward {value_key} + backward {value_key} (sequential stages)",
        "forward_ms": forward,
        "backward_ms": backward,
        value_key: forward + backward,
    }


def command_analyze(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    measurements_path = output_dir / "measurements.json"
    measurements = json.loads(measurements_path.read_text())
    raster_sass_text = (output_dir / "raster.sass").read_text()
    micro_sass_text = (output_dir / "microbench.sass").read_text()
    raster_functions = parse_sass_functions(raster_sass_text)
    micro_functions = parse_sass_functions(micro_sass_text)
    fwd_name, fwd_sass = select_sass_function(
        raster_functions, "rasterize_to_pixels_3dgs_fwd_sm120"
    )
    bwd_name, bwd_sass = select_sass_function(
        raster_functions, "rasterize_to_pixels_3dgs_bwd_sm120"
    )
    fwd_ncu = _ncu_kernel_summary(
        (output_dir / "ncu-forward-raw.csv").read_text(),
        (output_dir / "ncu-forward-details.csv").read_text(),
        fwd_sass,
        "rasterize_to_pixels_3dgs_fwd_sm120",
    )
    bwd_ncu = _ncu_kernel_summary(
        (output_dir / "ncu-backward-raw.csv").read_text(),
        (output_dir / "ncu-backward-details.csv").read_text(),
        bwd_sass,
        "rasterize_to_pixels_3dgs_bwd_sm120",
    )
    micro_ncu = _ncu_micro_summaries(
        (output_dir / "ncu-micro-raw.csv").read_text(),
        (output_dir / "ncu-micro-details.csv").read_text(),
        micro_functions,
        int(measurements["environment"]["sm_count"]),
    )
    probes = measurements["microbenchmarks"]["probes"]
    baseline = json.loads(Path("benchmark-results/baseline-analysis.json").read_text())
    selected_ceilings = baseline["roofline"]["selected_empirical_ceilings"]
    fp32_ceiling = selected_ceilings["ieee_fp32_tflops"]
    dram_ceiling = selected_ceilings["dram_gb_s_read_plus_write"]

    scenario_settings = {
        "optimistic": {
            "rate_quantile": "maximum",
            "launch_latency_quantile": "minimum",
            "contention_probe": "atomic_warp4",
            "barrier_probe": "barrier_balanced",
            "chain_mufu": False,
        },
        "central": {
            "rate_quantile": "median",
            "launch_latency_quantile": "median",
            "contention_probe": "atomic_warp8",
            "barrier_probe": "barrier_balanced",
            "chain_mufu": False,
        },
        "conservative": {
            "rate_quantile": "q05",
            "launch_latency_quantile": "q95",
            "contention_probe": "atomic_crosscta8",
            "barrier_probe": "barrier_skew8",
            "chain_mufu": True,
        },
    }
    raster_timings = {
        stage: measurements["raster"][stage]["timing_ms"]
        for stage in ("forward", "backward")
    }
    scenarios: dict[str, Any] = {}
    for scenario, settings in scenario_settings.items():
        stage_models = {
            "forward": _kernel_model(
                fwd_ncu, probes, stage="forward", fp32_ceiling_tflops=fp32_ceiling,
                dram_ceiling_gbs=dram_ceiling,
                current_wrapper_duration_ms=raster_timings["forward"]["median"],
                **settings,
            ),
            "backward": _kernel_model(
                bwd_ncu, probes, stage="backward", fp32_ceiling_tflops=fp32_ceiling,
                dram_ceiling_gbs=dram_ceiling,
                current_wrapper_duration_ms=raster_timings["backward"]["median"],
                **settings,
            ),
        }
        current = {
            "basis": "non-profiled CUDA-event wrapper timing",
            "forward_ms": raster_timings["forward"]["median"],
            "backward_ms": raster_timings["backward"]["median"],
        }
        current["combined_ms"] = current["forward_ms"] + current["backward_ms"]
        for stage in ("forward", "backward"):
            for model_key, value_key in (
                ("optimistic_l2_only_roofline", "lower_bound_ms"),
                (
                    "operation_aware_empirical_sustained_resource_target",
                    "target_ms",
                ),
            ):
                target = stage_models[stage][model_key][value_key]
                stage_models[stage][model_key]["efficiency_percent"] = (
                    100.0 * target / current[f"{stage}_ms"]
                )
                stage_models[stage][model_key]["residual_relative_to_target_percent"] = (
                    100.0 * (current[f"{stage}_ms"] / target - 1.0)
                )
        combined_old = _combine_stage_models(
            stage_models, "optimistic_l2_only_roofline", "lower_bound_ms"
        )
        combined_operation = _combine_stage_models(
            stage_models,
            "operation_aware_empirical_sustained_resource_target",
            "target_ms",
        )
        for combined, value_key in (
            (combined_old, "lower_bound_ms"),
            (combined_operation, "target_ms"),
        ):
            target = combined[value_key]
            combined["efficiency_percent"] = 100.0 * target / current["combined_ms"]
            combined["residual_relative_to_target_percent"] = 100.0 * (
                current["combined_ms"] / target - 1.0
            )
        scenarios[scenario] = {
            "settings": settings,
            "current": current,
            "stages": stage_models,
            "combined": {
                "optimistic_l2_only_roofline": combined_old,
                "operation_aware_empirical_sustained_resource_target": (
                    combined_operation
                ),
            },
        }

    current_q05 = (
        raster_timings["forward"]["q05"] + raster_timings["backward"]["q05"]
    )
    current_q95 = (
        raster_timings["forward"]["q95"] + raster_timings["backward"]["q95"]
    )
    scenario_resource_targets = [
        scenario["combined"][
            "operation_aware_empirical_sustained_resource_target"
        ][
            "target_ms"
        ]
        for scenario in scenarios.values()
    ]
    target_low = min(scenario_resource_targets)
    target_high = min(max(scenario_resource_targets), current_q05)
    criteria = criteria_from_ranges(
        resource_target_low_ms=target_low,
        current_q95_ms=current_q95,
        current_q05_ms=current_q05,
        resource_target_high_ms=target_high,
    )
    central_operation = scenarios["central"]["combined"][
        "operation_aware_empirical_sustained_resource_target"
    ]["target_ms"]
    central_old = scenarios["central"]["combined"][
        "optimistic_l2_only_roofline"
    ]["lower_bound_ms"]

    expected_micro = {
        "kc_reduction_ilp9": {"SHFL": 45, "FADD": 45},
        "kc_reduction_chain": {"SHFL": 5, "FADD": 5},
        "kc_redux_max_chain": {"REDUX": 1},
        "kc_atomic_redg": {"REDG": 9},
        "kc_barrier": {"BAR": 1},
        "kc_shared_lds128": {"LDS.128": 2},
        "kc_shared_sts128": {"STS.128": 2},
        "kc_mufu_ex2_ilp4": {"MUFU.EX2": 4},
        "kc_mufu_ex2_chain": {"MUFU.EX2": 1},
        "kc_mufu_rcp_ilp4": {"MUFU.RCP": 4},
        "kc_mufu_rcp_chain": {"MUFU.RCP": 1},
        "kc_empty_grid": {},
    }
    micro_proof: dict[str, Any] = {}
    for substring, expected in expected_micro.items():
        name, instructions = select_sass_function(micro_functions, substring)
        micro_proof[name] = _sass_proof(instructions, expected)

    commands = json.loads((output_dir / "commands.json").read_text())
    ncu_profiled_current = {
        "forward_ms": fwd_ncu["duration_ms"]["median"],
        "backward_ms": bwd_ncu["duration_ms"]["median"],
    }
    ncu_profiled_current["combined_ms"] = (
        ncu_profiled_current["forward_ms"]
        + ncu_profiled_current["backward_ms"]
    )
    profile_comparable: dict[str, Any] = {"current": ncu_profiled_current}
    for model_key, value_key in (
        ("optimistic_l2_only_roofline", "lower_bound_ms"),
        (
            "operation_aware_empirical_sustained_resource_target",
            "target_ms",
        ),
    ):
        stage_result: dict[str, Any] = {}
        for stage in ("forward", "backward"):
            target = scenarios["central"]["stages"][stage][model_key][value_key]
            stage_result[stage] = {
                value_key: target,
                "efficiency_percent": (
                    100.0 * target / ncu_profiled_current[f"{stage}_ms"]
                ),
            }
        combined_target = (
            stage_result["forward"][value_key]
            + stage_result["backward"][value_key]
        )
        profile_comparable[model_key] = {
            "stages": stage_result,
            "combined": {
                value_key: combined_target,
                "efficiency_percent": (
                    100.0 * combined_target / ncu_profiled_current["combined_ms"]
                ),
                "residual_relative_to_target_percent": 100.0
                * (ncu_profiled_current["combined_ms"] / combined_target - 1.0),
            },
        }
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "analysis.json" and not path.name.endswith(".ncu-rep"):
            artifacts.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "operation-aware-kernel-ceiling",
        "generated_at_utc": _utc_now(),
        "repository": {
            "commit": _git(["rev-parse", "HEAD"]),
            "starting_commit": "fc88cbd32a08bb5776313bceee4ce591d7d91fbc",
            "promoted_work_reverted": False,
            "worktree_status_porcelain": _git(["status", "--porcelain"]),
        },
        "environment": measurements["environment"],
        "case": measurements["raster"]["case"],
        "exact_commands": commands,
        "methodology": {
            "kernel_resource_target_equation": "max(FP32, DRAM, L2, shared, MUFU, barrier, reduction issue, reduction dependency, REDG atomic, launch/tail)",
            "combined_resource_target_equation": "forward max-target + backward max-target because the two kernels execute sequentially",
            "overlap_rule": "Independent resources within one kernel are not summed. EX2+RCP and LDS+STS are summed only within their single shared issue path; they are exclusive demands on that resource.",
            "current_latency_basis": "Non-profiled CUDA-event wrapper samples. Forward contains exactly one promoted raster launch. Backward includes the promoted launch plus four required output zero-initializations, so its measured latency is a conservative upper bound for the kernel and the reported efficiency is a lower bound.",
            "ncu_latency_use": "NCU duration is retained only for profile-comparable reporting and deriving the NCU L2 sustained peak; replay-perturbed duration is not the primary current latency.",
            "ncu_unit_normalization": "Time metrics are normalized to milliseconds and byte metrics to decimal bytes while retaining raw values and units; unknown units fail analysis.",
            "same_algorithm_definition": "same candidate traversal, staging batches, synchronization, warp reductions, gradient atomics, and transcendental operations as the promoted CUDA/SASS kernels",
            "empirical_resource_target_interpretation": "The operation-aware max term is an empirical sustained resource target from independent probe rates. It is neither a physical lower latency bound nor an achievable direct-PTX claim, because the rates need not be simultaneously attainable by the production kernel.",
            "residual_definition": "100 * (current exact-kernel latency / empirical sustained resource target - 1)",
        },
        "empirical_reference_ceilings": {
            "fp32_tflops": fp32_ceiling,
            "dram_gb_s": dram_ceiling,
            "source": "benchmark-results/baseline-analysis.json",
            "source_sha256": _sha256(Path("benchmark-results/baseline-analysis.json")),
        },
        "promoted_kernel_ncu": {"forward": fwd_ncu, "backward": bwd_ncu},
        "microbenchmark_ncu_dynamic_validation": micro_ncu,
        "sass_proof": {
            "forward": {
                "function": fwd_name,
                **_sass_proof(
                    fwd_sass,
                    {"BAR.RED": 1, "BAR.SYNC": 1, "LDS.128": 2, "STS.128": 2, "MUFU.EX2": 1},
                ),
            },
            "backward": {
                "function": bwd_name,
                **_sass_proof(
                    bwd_sass,
                    {"BAR.SYNC": 2, "SHFL": 55, "REDG": 11, "REDUX": 1, "MUFU.EX2": 1, "MUFU.RCP": 1},
                ),
            },
            "microbenchmarks": micro_proof,
            "compiler_elimination_guard": "all probe outputs are materialized and checksummed; volatile inline PTX fixes MUFU opcodes; NCU dynamic counts validate execution",
        },
        "microbenchmark_throughput_and_raw_samples": measurements["microbenchmarks"],
        "non_profiled_raster_wrapper_timings": measurements["raster"],
        "sensitivity": {
            "scenarios": scenarios,
            "empirical_sustained_resource_target_combined_ms_range": [
                target_low,
                target_high,
            ],
            "current_non_profiled_wrapper_combined_ms_q05_q95": [current_q05, current_q95],
            "central_operation_aware_efficiency_percent": scenarios["central"]["combined"]["operation_aware_empirical_sustained_resource_target"]["efficiency_percent"],
            "central_residual_relative_to_resource_target_percent": scenarios["central"]["combined"]["operation_aware_empirical_sustained_resource_target"]["residual_relative_to_target_percent"],
        },
        "ceiling_distinctions": {
            "existing_official_optimistic_l2_only_combined_efficiency_percent": 18.48,
            "fresh_non_profiled_wrapper_optimistic_l2_only": scenarios["central"]["combined"]["optimistic_l2_only_roofline"],
            "operation_aware_empirical_sustained_resource_target_non_profiled_wrapper": scenarios["central"]["combined"]["operation_aware_empirical_sustained_resource_target"],
            "ncu_profile_comparable_not_primary_latency": profile_comparable,
            "possible_algorithm_changing_headroom": {
                "empirical_resource_target_over_l2_only_bound": central_operation / central_old,
                "interpretation": "Changing traversal/reduction/accumulation may remove operation-aware terms and approach the L2-only physical bound. Reducing L2 traffic or useful work could go further and is deliberately not quantified by this resource target.",
                "not_an_achievable_direct_ptx_claim": True,
            },
        },
        "criteria": criteria,
        "criteria_genuinely_established": (
            criteria["at_least_25_percent_of_ceiling_established"]
            and criteria["less_than_10_percent_residual_established"]
        ),
        "assumptions_and_limitations": [
            "NCU replay perturbs absolute latency. Its five exact-kernel launches provide dynamic-count stability and a bridge to the historical 18.48% convention, but the 100 non-profiled CUDA-event wrapper samples determine primary current efficiency and residual.",
            "The backward public wrapper performs four output zero-initializations before the promoted kernel. Treating the full wrapper latency as current kernel latency can only reduce reported efficiency and increase residual, so criterion conclusions remain conservative.",
            "Only probe rounds starting at least 90% of the maximum observed round-start SM clock and at or below 85 C feed throughput rates; rejected rounds remain in raw samples.",
            "Microbenchmarks match 256-thread CTAs and force the target CTA residency with dynamic shared memory; NCU records actual occupancy and register allocation.",
            "The central REDG probe uses eight warp leaders contending on each of nine addresses within a CTA, matching the absgrad=False backward accumulation pattern; four-warp and eight-CTA contention probes bound sensitivity.",
            "Balanced barrier throughput is central. An eight-dependent-FFMA half-warp skew probe is intentionally conservative and used only in sensitivity.",
            "InstructionStats reports dynamic warp-level opcode executions. Per-PC mapping separates MUFU.EX2/RCP and BAR variants using exact captured SASS.",
            "An empirical resource target can expose that a criterion is not established; it cannot prove an implementation can attain all resource rates at once.",
        ],
        "artifact_manifest_excluding_raw_ncu_reports": artifacts,
        "raw_ncu_reports_retained_untracked": [
            str(output_dir / f"ncu-{target}.ncu-rep")
            for target in ("forward", "backward", "micro")
        ],
    }
    output = output_dir / "analysis.json"
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operation-aware ceiling for the promoted RTX 5090 raster kernels"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    measure.add_argument("--seed", type=int, default=SEED)
    measure.add_argument("--thermal-warmup-seconds", type=float, default=8.0)
    measure.add_argument("--rounds", type=int, default=5)
    measure.add_argument("--samples-per-round", type=int, default=20)
    measure.add_argument("--repetitions-per-sample", type=int, default=5)
    measure.add_argument("--probe-warmup", type=int, default=5)
    measure.add_argument("--raster-warmup", type=int, default=20)
    measure.add_argument("--raster-samples", type=int, default=100)
    measure.set_defaults(function=command_measure)

    profile = subparsers.add_parser("profile-target")
    profile.add_argument("--target", choices=("forward", "backward", "micro"), required=True)
    profile.add_argument("--target-warmup", type=int, default=5)
    profile.add_argument("--target-launches", type=int, default=5)
    profile.set_defaults(function=command_profile_target)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analyze.set_defaults(function=command_analyze)

    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    run.add_argument("--thermal-warmup-seconds", type=float, default=8.0)
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--samples-per-round", type=int, default=20)
    run.add_argument("--repetitions-per-sample", type=int, default=5)
    run.add_argument("--raster-samples", type=int, default=100)
    run.add_argument("--ncu-raster-launches", type=int, default=5)
    run.set_defaults(function=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "rounds",
        "samples_per_round",
        "repetitions_per_sample",
        "raster_samples",
        "ncu_raster_launches",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
