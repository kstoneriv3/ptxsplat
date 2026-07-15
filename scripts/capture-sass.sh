#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy.cubin}"
OUTPUT="${2:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy.sass}"

fail() {
  printf 'capture-sass: %s\n' "$*" >&2
  exit 2
}

command -v cuobjdump >/dev/null 2>&1 || fail \
  "cuobjdump is required; run this script through ./scripts/docker-run.sh"
[[ -f "${INPUT}" ]] || fail \
  "input cubin not found: ${INPUT}; run scripts/ptx-assemble.sh first"

mkdir -p "$(dirname "${OUTPUT}")"
TMP_OUTPUT="${OUTPUT}.tmp.$$"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

cuobjdump --dump-sass "${INPUT}" >"${TMP_OUTPUT}"

[[ -s "${TMP_OUTPUT}" ]] || fail "cuobjdump generated an empty SASS listing"
grep -Fq 'Function : smoke_axpy' "${TMP_OUTPUT}" || fail \
  "SASS listing does not contain the smoke_axpy function"

mv "${TMP_OUTPUT}" "${OUTPUT}"
trap - EXIT
printf 'SASS: %s\n' "${OUTPUT}"
