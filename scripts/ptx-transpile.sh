#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy.ptx}"
OUTPUT="${2:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy_pyptx.py}"

fail() {
  printf 'ptx-transpile: %s\n' "$*" >&2
  exit 2
}

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
[[ -f "${INPUT}" ]] || fail \
  "input PTX not found: ${INPUT}; run scripts/ptx-codegen.sh first"

PYPTX_VERSION="$({ python3 - <<'PY'
import importlib.metadata

try:
    print(importlib.metadata.version("pyptx"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(2)
PY
} 2>/dev/null)" || fail \
  "PyPTX 0.1.1 is optional and not installed; install it in a development environment with 'python3 -m pip install pyptx==0.1.1'"

[[ "${PYPTX_VERSION}" == "0.1.1" ]] || fail \
  "expected PyPTX 0.1.1, found ${PYPTX_VERSION}"

mkdir -p "$(dirname "${OUTPUT}")"
TMP_OUTPUT="${OUTPUT}.tmp.$$"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

python3 -m pyptx.codegen \
  "${INPUT}" \
  --name smoke_axpy >"${TMP_OUTPUT}"

[[ -s "${TMP_OUTPUT}" ]] || fail "PyPTX generated an empty file"
grep -Fq '@kernel' "${TMP_OUTPUT}" || fail \
  "PyPTX output does not contain a kernel declaration"

mv "${TMP_OUTPUT}" "${OUTPUT}"
trap - EXIT
printf 'PyPTX generator: %s\n' "${OUTPUT}"
