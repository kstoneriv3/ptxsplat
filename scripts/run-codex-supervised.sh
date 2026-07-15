#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${PTXSPLAT_CODEX_MODEL:-gpt-5.6-sol}"
EFFORT="${PTXSPLAT_CODEX_EFFORT:-xhigh}"
RETRY_SECONDS="${PTXSPLAT_CODEX_RETRY_SECONDS:-1800}"
PROFILE="${PTXSPLAT_CODEX_PROFILE:-0}"
LOG_DIR="${PTXSPLAT_CODEX_LOG_DIR:-${SCRIPT_DIR}/../.bcodex}"
LOG_FILE="${LOG_DIR}/autonomous-codex.log"
ATTEMPT_LOG="${LOG_DIR}/autonomous-codex.attempt.log"
INITIAL_PROMPT="${*:-Continue implementing docs/ROADMAP.md end to end. Work autonomously through implementation, CUDA build, correctness tests, benchmarks, commits, and push. Never weaken correctness tolerances or switch to a lower-effort model when a usage limit is reached.}"
RESUME_PROMPT="Continue the task autonomously. Keep the pinned model and reasoning effort. If the previous request stopped at a usage limit, first inspect the current repository state and resume from the last verified step."

case "${RETRY_SECONDS}" in
  ''|*[!0-9]*)
    printf 'PTXSPLAT_CODEX_RETRY_SECONDS must be a non-negative integer\n' >&2
    exit 2
    ;;
esac

if [[ "${PROFILE}" != "0" && "${PROFILE}" != "1" ]]; then
  printf 'PTXSPLAT_CODEX_PROFILE must be 0 or 1\n' >&2
  exit 2
fi

DOCKER_ARGS=()
if [[ "${PROFILE}" == "1" ]]; then
  DOCKER_ARGS=(--profile)
fi

mkdir -p "${LOG_DIR}"

run_codex() {
  : >"${ATTEMPT_LOG}"
  "${SCRIPT_DIR}/docker-run.sh" "${DOCKER_ARGS[@]}" -- "$@" 2>&1 \
    | tee -a "${LOG_FILE}" "${ATTEMPT_LOG}"
  return "${PIPESTATUS[0]}"
}

usage_limit_reached() {
  grep -Eiq \
    'rate.?limit|usage.?limit|quota|too many requests|http[^0-9]*429|status[^0-9]*429' \
    "${ATTEMPT_LOG}"
}

printf '[%s] starting supervised Codex run with %s/%s\n' \
  "$(date --iso-8601=seconds)" "${MODEL}" "${EFFORT}" | tee -a "${LOG_FILE}"

run_codex codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --model "${MODEL}" \
  --config "model_reasoning_effort=\"${EFFORT}\"" \
  "${INITIAL_PROMPT}"
status=$?

while (( status != 0 )); do
  if ! usage_limit_reached; then
    printf '[%s] Codex failed for a reason other than a recognized usage limit; stopping\n' \
      "$(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
    exit "${status}"
  fi
  printf '[%s] Codex exited with status %d; retrying in %s seconds\n' \
    "$(date --iso-8601=seconds)" "${status}" "${RETRY_SECONDS}" | tee -a "${LOG_FILE}"
  sleep "${RETRY_SECONDS}"
  run_codex codex exec resume --last \
    --dangerously-bypass-approvals-and-sandbox \
    --model "${MODEL}" \
    --config "model_reasoning_effort=\"${EFFORT}\"" \
    "${RESUME_PROMPT}"
  status=$?
done

printf '[%s] Codex completed successfully\n' "$(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
