#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_ARGS=()
if [[ "${1:-}" == "--profile" ]]; then
  PROFILE_ARGS=(--profile)
  shift
fi

exec "${SCRIPT_DIR}/docker-run.sh" "${PROFILE_ARGS[@]}" -- codex --yolo "$@"
