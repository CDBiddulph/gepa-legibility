#!/bin/bash
# Generate metrics data for plot_paper_figures.ipynb
# Runs jobs with a configurable level of parallelism

# === CONFIGURATION ===
MAX_PARALLEL=2  # Number of jobs to run in parallel
# =====================

METRICS="proxy_reward true_reward hacking_rate prompt_verbalizes"
LOG_DIR="logs/metrics_generation"
mkdir -p "$LOG_DIR"

# Define all jobs as "name|path" pairs
JOBS=(
  "mcq|logs/mcq/hack-teacher=false/2025-12-23-00-02-21/"
  "psychosis_teacher_false|logs/psychosis/prompter-hack-teacher=false/2025-12-12-00-24-33/"
  "psychosis_teacher_true|logs/psychosis/prompter-hack=explicit-teacher=true/2026-01-01-11-42-57/"
  "wordchain_teacher_false|logs/wordchain/prompter-hack-teacher=false/2025-12-28-23-10-01"
  "wordchain_teacher_true|logs/wordchain/prompter-hack=explicit-teacher=true/2026-01-01-11-40-31"
)

# Track running jobs: PID -> job name
declare -A RUNNING_PIDS
NEXT_JOB=0
FAILED=0

start_job() {
  local job_spec="${JOBS[$NEXT_JOB]}"
  local name="${job_spec%%|*}"
  local path="${job_spec##*|}"

  python -m core.evaluation.progression "$path" --metrics $METRICS \
    > "$LOG_DIR/${name}.log" 2>&1 &

  local pid=$!
  RUNNING_PIDS[$pid]="$name"
  echo "[$(date '+%H:%M:%S')] Started: $name (PID $pid)"

  NEXT_JOB=$((NEXT_JOB + 1))
}

wait_for_one() {
  # Poll until one job finishes
  while true; do
    for pid in "${!RUNNING_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
        EXIT_CODE=$?
        FINISHED_PID=$pid
        return
      fi
    done
    sleep 1
  done
}

echo "Starting metrics generation at $(date)"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Total jobs: ${#JOBS[@]}"
echo "Logs: $LOG_DIR/"
echo ""

# Start initial batch
while [ ${#RUNNING_PIDS[@]} -lt $MAX_PARALLEL ] && [ $NEXT_JOB -lt ${#JOBS[@]} ]; do
  start_job
done

# Process jobs as they complete
while [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
  wait_for_one

  # Report completion
  name="${RUNNING_PIDS[$FINISHED_PID]}"
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] Completed: $name"
  else
    echo "[$(date '+%H:%M:%S')] FAILED: $name (exit code $EXIT_CODE)"
    FAILED=$((FAILED + 1))
  fi
  unset "RUNNING_PIDS[$FINISHED_PID]"

  # Start next job if any remain
  if [ $NEXT_JOB -lt ${#JOBS[@]} ]; then
    start_job
  fi
done

echo ""
echo "Completed at $(date)"
if [ $FAILED -eq 0 ]; then
  echo "All ${#JOBS[@]} jobs succeeded!"
else
  echo "$FAILED job(s) failed. Check logs in $LOG_DIR/"
  exit 1
fi
