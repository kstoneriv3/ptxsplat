#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy_pyptx.py}"
OUTPUT="${2:-${REPO_ROOT}/.bcodex/ptx/smoke_axpy_pyptx.ptx}"
KERNEL_NAME="${3:-smoke_axpy}"

fail() {
  printf 'ptx-emit: %s\n' "$*" >&2
  exit 2
}

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
[[ -f "${INPUT}" ]] || fail \
  "PyPTX source not found: ${INPUT}; run scripts/ptx-transpile.sh first"

PYPTX_VERSION="$({ python3 - <<'PY'
import importlib.metadata

try:
    print(importlib.metadata.version("pyptx"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(2)
PY
} 2>/dev/null)" || fail \
  "PyPTX 0.1.1 is optional and not installed; add its development directory to PYTHONPATH"

[[ "${PYPTX_VERSION}" == "0.1.1" ]] || fail \
  "expected PyPTX 0.1.1, found ${PYPTX_VERSION}"

mkdir -p "$(dirname "${OUTPUT}")"
TMP_OUTPUT="${OUTPUT}.tmp.$$"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

python3 - "${INPUT}" "${TMP_OUTPUT}" "${KERNEL_NAME}" <<'PY'
from pathlib import Path
import runpy
import sys

source_path, output_path, kernel_name = sys.argv[1:]
namespace = runpy.run_path(source_path)
candidate = namespace.get(kernel_name)
if candidate is None or not callable(getattr(candidate, "ptx", None)):
    raise SystemExit(f"PyPTX source does not define kernel {kernel_name!r}")
Path(output_path).write_text(candidate.ptx(), encoding="utf-8")
PY

[[ -s "${TMP_OUTPUT}" ]] || fail "PyPTX emitted an empty PTX file"
grep -Eq '^\.target[[:space:]]+sm_120([,[:space:]]|$)' "${TMP_OUTPUT}" || \
  fail "PyPTX output does not target sm_120"
grep -Fq ".entry ${KERNEL_NAME}(" "${TMP_OUTPUT}" || fail \
  "PyPTX output does not contain the ${KERNEL_NAME} entry point"

mv "${TMP_OUTPUT}" "${OUTPUT}"
trap - EXIT
printf 'Re-emitted PTX: %s\n' "${OUTPUT}"
