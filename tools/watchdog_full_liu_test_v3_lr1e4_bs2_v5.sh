#!/usr/bin/env bash
set -u

# Watchdog for:
# full_liu_test_v3_lr1e4_bs2_v5
#
# Behavior:
# - Restart training when process exits abnormally (non-zero exit code),
#   including killed cases like 137 (SIGKILL/OOM) and 143 (SIGTERM).
# - Stop restarting when training exits with code 0.

RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-10}"
MAX_RESTARTS="${MAX_RESTARTS:-0}" # 0 means unlimited
LOG_DIR="${LOG_DIR:-./saved/watchdog_logs}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/full_liu_test_v3_lr1e4_bs2_v5_watchdog.log"

WATCHDOG_STOP=0
CHILD_PID=""
RESTART_COUNT=0

on_stop() {
  WATCHDOG_STOP=1
  echo "[$(date '+%F %T')] Watchdog received stop signal." | tee -a "${LOG_FILE}"
  if [[ -n "${CHILD_PID}" ]]; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" 2>/dev/null || true
  fi
  exit 0
}

trap on_stop INT TERM

while true; do
  if [[ "${WATCHDOG_STOP}" -eq 1 ]]; then
    break
  fi

  START_TS="$(date '+%F %T')"
  echo "[${START_TS}] Starting training (restart_count=${RESTART_COUNT})..." | tee -a "${LOG_FILE}"

  (
    export CACHE_DIR="/root/autodl-tmp/cached"
    OMP_NUM_THREADS=3 python "train.py" \
      -cn=liunet_cached_ep100_bs6_warm10_lr2e4_1e5_fullval_10_5_2 \
      writer.run_name='full_liu_test_v3_lr1e4_bs2_v5' \
      trainer.max_grad_norm=1 \
      dataloader.batch_size=2 \
      optimizer.lr=1e-4 \
      trainer.warmup.enabled=false \
      lr_scheduler.eta_min=1e-4
  ) >>"${LOG_FILE}" 2>&1 &

  CHILD_PID="$!"
  wait "${CHILD_PID}"
  EXIT_CODE="$?"
  CHILD_PID=""

  END_TS="$(date '+%F %T')"
  echo "[${END_TS}] Training exited with code ${EXIT_CODE}." | tee -a "${LOG_FILE}"

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "[${END_TS}] Training completed successfully. Watchdog exits." | tee -a "${LOG_FILE}"
    exit 0
  fi

  RESTART_COUNT="$((RESTART_COUNT + 1))"
  if [[ "${MAX_RESTARTS}" -gt 0 && "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]]; then
    echo "[${END_TS}] Reached MAX_RESTARTS=${MAX_RESTARTS}. Watchdog exits." | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "[${END_TS}] Restarting in ${RESTART_DELAY_SECONDS}s..." | tee -a "${LOG_FILE}"
  sleep "${RESTART_DELAY_SECONDS}"
done

