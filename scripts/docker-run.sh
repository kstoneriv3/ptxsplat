#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PTXSPLAT_IMAGE:-360-video-gs-dev:latest}"
STATE_ROOT="${PTXSPLAT_STATE_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/ptxsplat}"
PROFILE=false

if [[ "${1:-}" == "--profile" ]]; then
  PROFILE=true
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  set -- bash
fi

mkdir -p "${STATE_ROOT}/home" "${REPO_ROOT}/.bcodex/torch_extensions"
docker image inspect "${IMAGE}" >/dev/null
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

CAP_ARGS=()
SETPRIV_CAPS=()
if [[ "${PROFILE}" == true ]]; then
  CAP_ARGS=(--cap-add SYS_ADMIN)
  SETPRIV_CAPS=(--inh-caps=+sys_admin --ambient-caps=+sys_admin)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
  --gpus 'device=0' \
  --shm-size 12g \
  "${CAP_ARGS[@]}" \
  --entrypoint /bin/bash \
  -e HOME=/ptxsplat-state/home \
  -e PTXSPLAT_DOCKER_IMAGE="${IMAGE}" \
  -e PTXSPLAT_DOCKER_IMAGE_ID="${IMAGE_ID}" \
  -e TORCH_EXTENSIONS_DIR=/workspace/.bcodex/torch_extensions \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "${REPO_ROOT}:/workspace" \
  -v "${STATE_ROOT}:/ptxsplat-state" \
  -w /workspace \
  "${IMAGE}" \
  -c 'set -e; mkdir -p /ptxsplat-state/home/.codex; chmod 700 /ptxsplat-state /ptxsplat-state/home /ptxsplat-state/home/.codex; exec /usr/bin/setpriv "$@"' \
  ptxsplat-entrypoint \
  --reuid="$(id -u)" --regid="$(id -g)" --clear-groups \
  "${SETPRIV_CAPS[@]}" -- "$@"
