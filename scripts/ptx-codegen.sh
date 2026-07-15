#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-${REPO_ROOT}/ptx/smoke/smoke_axpy.cu}"
OUTPUT="${2:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy.ptx}"

fail() {
  printf 'ptx-codegen: %s\n' "$*" >&2
  exit 2
}

command -v nvcc >/dev/null 2>&1 || fail \
  "nvcc is required; run this script through ./scripts/docker-run.sh"
[[ -f "${SOURCE}" ]] || fail "source file not found: ${SOURCE}"

mkdir -p "$(dirname "${OUTPUT}")"
TMP_OUTPUT="${OUTPUT}.tmp.$$"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

nvcc \
  --ptx \
  --gpu-architecture=compute_120 \
  --std=c++17 \
  --optimize=3 \
  --generate-line-info \
  --output-file "${TMP_OUTPUT}" \
  "${SOURCE}"

grep -Eq '^\.target[[:space:]]+sm_120([,[:space:]]|$)' "${TMP_OUTPUT}" || \
  fail "nvcc output does not target sm_120: ${TMP_OUTPUT}"
grep -Fq '.entry smoke_axpy(' "${TMP_OUTPUT}" || \
  fail "nvcc output does not contain the smoke_axpy entry point"

mv "${TMP_OUTPUT}" "${OUTPUT}"
trap - EXIT
printf 'PTX: %s\n' "${OUTPUT}"
