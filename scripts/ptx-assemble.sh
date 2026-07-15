#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy.ptx}"
OUTPUT="${2:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy.cubin}"

fail() {
  printf 'ptx-assemble: %s\n' "$*" >&2
  exit 2
}

command -v ptxas >/dev/null 2>&1 || fail \
  "ptxas is required; run this script through ./scripts/docker-run.sh"
[[ -f "${INPUT}" ]] || fail \
  "input PTX not found: ${INPUT}; run scripts/ptx-codegen.sh first"
grep -Eq '^\.target[[:space:]]+sm_120([,[:space:]]|$)' "${INPUT}" || fail \
  "input PTX does not declare target sm_120: ${INPUT}"

mkdir -p "$(dirname "${OUTPUT}")"
TMP_OUTPUT="${OUTPUT}.tmp.$$"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

ptxas \
  --gpu-name=sm_120 \
  --verbose \
  --warn-on-spills \
  --generate-line-info \
  --output-file "${TMP_OUTPUT}" \
  "${INPUT}"

[[ -s "${TMP_OUTPUT}" ]] || fail "ptxas generated an empty cubin"
mv "${TMP_OUTPUT}" "${OUTPUT}"
trap - EXIT
printf 'CUBIN: %s\n' "${OUTPUT}"
