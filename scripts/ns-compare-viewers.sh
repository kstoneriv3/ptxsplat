#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PTXSPLAT_IMAGE:-360-video-gs-dev:latest}"
STATE_ROOT="${PTXSPLAT_STATE_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/ptxsplat}"
UID_VALUE="$(id -u)"
GID_VALUE="$(id -g)"

UPSTREAM_NAME="ptxsplat-ns-compare-upstream"
PTXSPLAT_NAME="ptxsplat-ns-compare-sm120"
UPSTREAM_CONFIG="/workspace/results/ns-compare/upstream/tiny-synthetic/splatfacto/matched-1000/config.yml"
PTXSPLAT_CONFIG="/workspace/results/ns-compare/ptxsplat/tiny-synthetic/splatfacto/matched-1000/config.yml"

usage() {
  cat <<'EOF'
Usage: scripts/ns-compare-viewers.sh {up|down|status|logs} [upstream|ptxsplat|all]

Runs the two Nerfstudio comparison viewers with stable container names:
  upstream: http://localhost:7007
  ptxsplat: http://localhost:7008

The launcher scopes TORCHDYNAMO_DISABLE=1 to these viewer processes only.
EOF
}

viewer_names() {
  case "${1:-all}" in
    upstream) printf '%s\n' "${UPSTREAM_NAME}" ;;
    ptxsplat|sm120) printf '%s\n' "${PTXSPLAT_NAME}" ;;
    all) printf '%s\n%s\n' "${UPSTREAM_NAME}" "${PTXSPLAT_NAME}" ;;
    *) usage >&2; exit 2 ;;
  esac
}

ensure_runtime_dirs() {
  mkdir -p "${STATE_ROOT}/home" "${REPO_ROOT}/.bcodex/torch_extensions"
}

image_id() {
  docker image inspect --format '{{.Id}}' "${IMAGE}"
}

remove_container() {
  local name="$1"
  if docker container inspect "${name}" >/dev/null 2>&1; then
    docker rm -f "${name}" >/dev/null
  fi
}

run_viewer() {
  local name="$1"
  local pythonpath="$2"
  local backend="$3"
  local config="$4"
  local port="$5"
  local image_digest="$6"
  local backend_env=()
  if [[ -n "${backend}" ]]; then
    backend_env=(-e "PTXSPLAT_BACKEND=${backend}")
  fi

  docker run -d \
    --name "${name}" \
    --gpus 'device=0' \
    --network host \
    --shm-size 12g \
    --entrypoint /bin/bash \
    -e HOME=/ptxsplat-state/home \
    -e USER=ptxsplat \
    -e LOGNAME=ptxsplat \
    -e USERNAME=ptxsplat \
    -e PYTHONPATH="${pythonpath}" \
    "${backend_env[@]}" \
    -e PTXSPLAT_DOCKER_IMAGE="${IMAGE}" \
    -e PTXSPLAT_DOCKER_IMAGE_ID="${image_digest}" \
    -e TORCH_EXTENSIONS_DIR=/workspace/.bcodex/torch_extensions \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "${REPO_ROOT}:/workspace" \
    -v "${STATE_ROOT}:/ptxsplat-state" \
    -w /workspace \
    "${IMAGE}" \
    -c 'set -e; mkdir -p /ptxsplat-state/home/.codex; chmod 700 /ptxsplat-state /ptxsplat-state/home /ptxsplat-state/home/.codex; exec /usr/bin/setpriv "$@"' \
    ptxsplat-ns-compare-entrypoint \
    --reuid="${UID_VALUE}" --regid="${GID_VALUE}" --clear-groups -- \
    env TORCHDYNAMO_DISABLE=1 \
    ns-viewer --load-config "${config}" --viewer.websocket-port "${port}" --viewer.websocket-host 0.0.0.0
}

up() {
  ensure_runtime_dirs
  docker image inspect "${IMAGE}" >/dev/null
  local digest
  digest="$(image_id)"

  remove_container "${UPSTREAM_NAME}"
  remove_container "${PTXSPLAT_NAME}"

  run_viewer \
    "${UPSTREAM_NAME}" \
    "/workspace/.bcodex/gsplat-1.5.3" \
    "" \
    "${UPSTREAM_CONFIG}" \
    "7007" \
    "${digest}"
  run_viewer \
    "${PTXSPLAT_NAME}" \
    "/workspace/compat/gsplat_overload:/workspace" \
    "sm120" \
    "${PTXSPLAT_CONFIG}" \
    "7008" \
    "${digest}"
}

down() {
  local name
  while IFS= read -r name; do
    remove_container "${name}"
  done < <(viewer_names "${1:-all}")
}

status() {
  docker ps -a \
    --filter "name=^/${UPSTREAM_NAME}$" \
    --filter "name=^/${PTXSPLAT_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Command}}'
}

logs() {
  local name
  while IFS= read -r name; do
    printf '==> %s\n' "${name}"
    docker logs --tail "${PTXSPLAT_VIEWER_LOG_LINES:-120}" "${name}"
  done < <(viewer_names "${1:-all}")
}

case "${1:-}" in
  up) up ;;
  down) down "${2:-all}" ;;
  status) status ;;
  logs) logs "${2:-all}" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
