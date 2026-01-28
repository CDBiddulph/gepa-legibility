#!/bin/bash
# Generate metrics data for plot_paper_figures.ipynb
# Runs jobs with a configurable level of parallelism, at the sweep config level
#
# Usage:
#   ./generate_metrics.sh        # uses default version from experiments_config.yaml
#   ./generate_metrics.sh v1     # uses specific version

# === CONFIGURATION ===
MAX_PARALLEL=3  # Number of jobs to run in parallel
VERSION="${1:-}"  # Empty means use default from config
# =====================

# Determine version name for log directory
USED_VERSION=$(python -c "
from core.experiment_utils import load_experiments_config
import yaml
from pathlib import Path
config_path = Path('experiments_config.yaml')
raw = yaml.safe_load(open(config_path))
version = '$VERSION' or raw['default_version']
print(version)
")

# Create timestamped log directory
TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
LOG_DIR="logs/metrics_generation/${USED_VERSION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Load jobs from experiments_config.yaml via Python
# Outputs sweep config directories: "name|sweep_path|metrics|quick_mode"
JOBS_OUTPUT=$(python -c "
from core.experiment_utils import load_experiments_config
from pathlib import Path
version = '$VERSION' or None
config = load_experiments_config(version)

# Define what each experiment type needs based on notebook CONFIGS
EXPERIMENT_REQUIREMENTS = {
    # (metrics, quick_mode)
    # verbalizes_* = CoT verbalization (instructions + CoT)
    # verbalizes_prompt_* = prompt verbalization (instructions only)
    'teacher_true': ('proxy_reward true_reward hacking_rate verbalizes_prompt_yes verbalizes_yes', False),
    'teacher_false': ('proxy_reward true_reward hacking_rate verbalizes_prompt_yes verbalizes_yes', False),
    'baseline_only': ('hacking_rate verbalizes_yes', True),
    'tinker': ('proxy_reward hacking_rate verbalizes_yes verbalizes_prompt_yes', True),
    'tinker_ckpts': ('hacking_rate verbalizes_yes verbalizes_prompt_yes', True),
}

for env, experiments in config.items():
    for exp_type, exp_path in experiments.items():
        if exp_type not in EXPERIMENT_REQUIREMENTS:
            continue  # skip unused experiment types
        metrics, quick_mode = EXPERIMENT_REQUIREMENTS[exp_type]
        quick_flag = '--quick-mode' if quick_mode else ''

        # Enumerate sweep config directories within each experiment
        exp_path = Path(exp_path)
        if not exp_path.exists():
            print(f'# Warning: {exp_path} does not exist', file=__import__('sys').stderr)
            continue

        # Find all sweep config directories (contain */detailed_results.json)
        for sweep_dir in sorted(exp_path.iterdir()):
            if not sweep_dir.is_dir():
                continue
            if not list(sweep_dir.glob('*/detailed_results.json')):
                continue
            name = f'{env}_{exp_type}_{sweep_dir.name}'
            print(f'{name}|{sweep_dir}|{metrics}|{quick_flag}')
")

# Convert to array, filtering empty lines
readarray -t JOBS < <(echo "$JOBS_OUTPUT" | grep -v '^$')

# Track running jobs: PID -> job name
declare -A RUNNING_PIDS
NEXT_JOB=0
FAILED=0

start_job() {
  local job_spec="${JOBS[$NEXT_JOB]}"
  IFS='|' read -r name sweep_path metrics quick_flag <<< "$job_spec"

  python -m core.evaluation.progression "$sweep_path" --metrics $metrics $quick_flag \
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
echo "Config version: $USED_VERSION"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Total jobs: ${#JOBS[@]} (sweep configs)"
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
