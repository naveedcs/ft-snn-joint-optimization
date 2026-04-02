#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/scripts/matched_ann_control_ftt.py"
WAIT_FOR_COMPLETION=false

if [[ "${1:-}" == "--wait" ]]; then
  WAIT_FOR_COMPLETION=true
elif [[ "${1:-}" != "" ]]; then
  echo "Usage: $0 [--wait]" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Runner not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
RANDOM_SEED="${RANDOM_SEED:-52}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/baf-datasets}"
DEFAULT_DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-auto}"

declare -a TAGS=("Base" "VI" "VII" "VIII" "VIV" "VV")

GPU_ASSIGNMENTS_RAW="${GPU_ASSIGNMENTS:-0,1,2,0,1,2}"
IFS=',' read -r -a GPU_ASSIGNMENTS <<< "${GPU_ASSIGNMENTS_RAW}"
if [[ "${#GPU_ASSIGNMENTS[@]}" -eq 1 ]]; then
  single_assignment="${GPU_ASSIGNMENTS[0]}"
  GPU_ASSIGNMENTS=()
  for _ in "${TAGS[@]}"; do
    GPU_ASSIGNMENTS+=("${single_assignment}")
  done
fi
if [[ "${#GPU_ASSIGNMENTS[@]}" -ne "${#TAGS[@]}" ]]; then
  echo "GPU_ASSIGNMENTS must have 1 or ${#TAGS[@]} comma-separated entries." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/parallel_runs_matched_ann_control/${TIMESTAMP}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PID_FILE="${RUN_ROOT}/pids.tsv"
printf "pid\ttag\tdevice\tlog\n" > "${PID_FILE}"

launch_run() {
  local tag="$1"
  local device_slot="$2"
  local log_path="$3"

  if [[ "${device_slot}" == "cpu" ]]; then
    nohup env \
      PYTHONUNBUFFERED=1 \
      EXPERIMENT_TAG="${tag}" \
      RANDOM_SEED="${RANDOM_SEED}" \
      DEVICE_OVERRIDE="cpu" \
      RESULTS_DIR="${RESULTS_DIR}" \
      DATASET_DIR="${DATASET_DIR}" \
      bash -c '
        cd "'"${ROOT_DIR}"'"
        exec "'"${PYTHON_BIN}"'" "'"${SCRIPT_PATH}"'"
      ' > "${log_path}" 2>&1 < /dev/null &
  else
    nohup env \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="${device_slot}" \
      CUDA_DEVICE_INDICES="0" \
      PREFERRED_GPU_INDEX="0" \
      EXPERIMENT_TAG="${tag}" \
      RANDOM_SEED="${RANDOM_SEED}" \
      DEVICE_OVERRIDE="${DEFAULT_DEVICE_OVERRIDE}" \
      RESULTS_DIR="${RESULTS_DIR}" \
      DATASET_DIR="${DATASET_DIR}" \
      bash -c '
        cd "'"${ROOT_DIR}"'"
        exec "'"${PYTHON_BIN}"'" "'"${SCRIPT_PATH}"'"
      ' > "${log_path}" 2>&1 < /dev/null &
  fi
  LAUNCHED_PID="$!"
}

echo "Launching matched-ANN control runs..."
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  device_slot="${GPU_ASSIGNMENTS[$i]}"
  log_path="${LOG_DIR}/${tag}.log"
  launch_run "${tag}" "${device_slot}" "${log_path}"
  pid="${LAUNCHED_PID}"
  printf "%s\t%s\t%s\t%s\n" "${pid}" "${tag}" "${device_slot}" "${log_path}" >> "${PID_FILE}"
  echo "  started tag=${tag} device=${device_slot} pid=${pid}"
done

echo
echo "Run directory: ${RUN_ROOT}"
echo "PID file: ${PID_FILE}"
echo "Watch logs: tail -f ${LOG_DIR}/*.log"

if [[ "${WAIT_FOR_COMPLETION}" == "true" ]]; then
  echo
  echo "Waiting for all runs to finish..."
  FAILED_RUNS=0
  TOTAL_RUNS="${#TAGS[@]}"
  COMPLETED_RUNS=0

  declare -A PID_TAG
  declare -A PID_DEVICE
  declare -A PID_LOG
  declare -A PID_DONE

  while IFS=$'\t' read -r pid tag device_slot log_path; do
    if [[ "${pid}" == "pid" ]]; then
      continue
    fi
    PID_TAG["${pid}"]="${tag}"
    PID_DEVICE["${pid}"]="${device_slot}"
    PID_LOG["${pid}"]="${log_path}"
    PID_DONE["${pid}"]=0
  done < "${PID_FILE}"

  while [[ "${COMPLETED_RUNS}" -lt "${TOTAL_RUNS}" ]]; do
    for pid in "${!PID_DONE[@]}"; do
      if [[ "${PID_DONE[${pid}]}" -eq 1 ]]; then
        continue
      fi
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi

      if wait "${pid}"; then
        status_code=0
      else
        status_code=$?
      fi

      PID_DONE["${pid}"]=1
      COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
      if [[ "${status_code}" -ne 0 ]]; then
        FAILED_RUNS=$((FAILED_RUNS + 1))
      fi
      echo "  completed tag=${PID_TAG[${pid}]} device=${PID_DEVICE[${pid}]} pid=${pid} status=${status_code} (log: ${PID_LOG[${pid}]})"
    done
    sleep 2
  done

  if [[ "${FAILED_RUNS}" -gt 0 ]]; then
    echo "Finished with failures: ${FAILED_RUNS} run(s) failed."
    exit 1
  fi

  echo "Finished successfully: all 6 runs completed."
fi
