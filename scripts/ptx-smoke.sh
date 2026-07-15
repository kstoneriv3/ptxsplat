#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PTXSPLAT_PTX_OUTPUT_DIR:-${REPO_ROOT}/.bcodex/ptx}"
WITH_PYPTX=false

if [[ "${1:-}" == "--with-pyptx" ]]; then
  WITH_PYPTX=true
  shift
fi
if [[ $# -ne 0 ]]; then
  printf 'usage: %s [--with-pyptx]\n' "$0" >&2
  exit 2
fi

fail() {
  printf 'ptx-smoke: %s\n' "$*" >&2
  exit 2
}

command -v nvidia-smi >/dev/null 2>&1 || fail \
  "nvidia-smi is required; run this script through ./scripts/docker-run.sh"

COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
  | head -n 1 | tr -d '[:space:]')"
[[ "${COMPUTE_CAP}" == "12.0" ]] || fail \
  "expected an RTX 5090-class compute capability 12.0 GPU, found ${COMPUTE_CAP:-unknown}"

mkdir -p "${OUTPUT_DIR}"
PTX="${OUTPUT_DIR}/smoke_axpy.ptx"
CUBIN="${OUTPUT_DIR}/smoke_axpy.cubin"
SASS="${OUTPUT_DIR}/smoke_axpy.sass"

"${REPO_ROOT}/scripts/ptx-codegen.sh" \
  "${REPO_ROOT}/ptx/smoke/smoke_axpy.cu" "${PTX}"

if [[ "${WITH_PYPTX}" == true ]]; then
  NVCC_CUBIN="${OUTPUT_DIR}/smoke_axpy_nvcc.cubin"
  NVCC_SASS="${OUTPUT_DIR}/smoke_axpy_nvcc.sass"
  "${REPO_ROOT}/scripts/ptx-assemble.sh" "${PTX}" "${NVCC_CUBIN}"
  "${REPO_ROOT}/scripts/capture-sass.sh" "${NVCC_CUBIN}" "${NVCC_SASS}"

  PYPTX_SOURCE="${OUTPUT_DIR}/smoke_axpy_pyptx.py"
  PYPTX_PTX="${OUTPUT_DIR}/smoke_axpy_pyptx.ptx"
  "${REPO_ROOT}/scripts/ptx-transpile.sh" \
    "${PTX}" "${PYPTX_SOURCE}"
  "${REPO_ROOT}/scripts/ptx-emit.sh" \
    "${PYPTX_SOURCE}" "${PYPTX_PTX}" smoke_axpy
  PTX="${PYPTX_PTX}"
else
  printf 'PyPTX transpilation: skipped (pass --with-pyptx to enable)\n'
fi

"${REPO_ROOT}/scripts/ptx-assemble.sh" "${PTX}" "${CUBIN}"
"${REPO_ROOT}/scripts/capture-sass.sh" "${CUBIN}" "${SASS}"

if [[ "${WITH_PYPTX}" == true ]]; then
  cmp -s "${NVCC_SASS}" "${SASS}" || fail \
    "PyPTX re-emission changed SASS; inspect ${NVCC_SASS} and ${SASS}"
  printf 'PyPTX/NVCC SASS match: PASS\n'
fi

printf 'sm_120 PTX smoke check: PASS (compute capability %s)\n' "${COMPUTE_CAP}"
