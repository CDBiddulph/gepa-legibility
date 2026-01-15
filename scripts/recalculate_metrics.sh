#!/bin/bash
# Script to recalculate specific metrics for given directories
#
# Usage:
#   ./scripts/recalculate_metrics.sh --metrics metric1 metric2 [--num-threads N] -- dir1 dir2 dir3
#
# Example:
#   ./scripts/recalculate_metrics.sh --metrics prompt_verbalizes cot_verbalizes --num-threads 8 -- logs/psychosis/run1 logs/psychosis/run2

set -e

# Parse arguments
METRICS=()
DIRS=()
NUM_THREADS=""
parsing_metrics=false
parsing_dirs=false

for arg in "$@"; do
    if [[ "$arg" == "--metrics" ]]; then
        parsing_metrics=true
        parsing_dirs=false
        continue
    elif [[ "$arg" == "--num-threads" ]]; then
        parsing_metrics=false
        parsing_dirs=false
        # Next arg will be the thread count
        NUM_THREADS="next"
        continue
    elif [[ "$arg" == "--" ]]; then
        parsing_metrics=false
        parsing_dirs=true
        continue
    fi

    if [[ "$NUM_THREADS" == "next" ]]; then
        NUM_THREADS="$arg"
        continue
    fi

    if $parsing_metrics; then
        METRICS+=("$arg")
    elif $parsing_dirs; then
        DIRS+=("$arg")
    fi
done

# Validate arguments
if [[ ${#METRICS[@]} -eq 0 ]]; then
    echo "ERROR: No metrics specified. Use --metrics metric1 metric2 ..."
    echo "Usage: $0 --metrics metric1 metric2 [--num-threads N] -- dir1 dir2 dir3"
    exit 1
fi

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No directories specified. Use -- dir1 dir2 ..."
    echo "Usage: $0 --metrics metric1 metric2 [--num-threads N] -- dir1 dir2 dir3"
    exit 1
fi

# Validate that metrics are real metrics
VALID_METRICS=$(python -c "from core.metrics import METRICS; print(' '.join(METRICS.keys()))")
for metric in "${METRICS[@]}"; do
    if ! echo "$VALID_METRICS" | grep -qw "$metric"; then
        echo "ERROR: Unknown metric '$metric'"
        echo "Valid metrics are: $VALID_METRICS"
        exit 1
    fi
done

# Build num-threads argument if specified
THREADS_ARG=""
if [[ -n "$NUM_THREADS" && "$NUM_THREADS" != "next" ]]; then
    THREADS_ARG="--num-threads $NUM_THREADS"
fi

echo "=== Recalculating metrics ==="
echo "Metrics: ${METRICS[*]}"
echo "Directories: ${DIRS[*]}"
if [[ -n "$THREADS_ARG" ]]; then
    echo "Threads: $NUM_THREADS"
fi
echo ""

# Validate directories have evaluations data
echo "=== Validating directories ==="
for dir in "${DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: Directory does not exist: $dir"
        exit 1
    fi
    eval_dirs=$(find "$dir" -type d -name "evaluations" 2>/dev/null || true)
    if [[ -z "$eval_dirs" ]]; then
        echo "ERROR: No evaluations directories found in $dir"
        exit 1
    fi
    count=$(echo "$eval_dirs" | wc -l)
    echo "Found $count evaluations directories in $dir"
done

# Delete existing metric cache files for specified metrics
echo ""
echo "=== Deleting existing metric caches ==="
for dir in "${DIRS[@]}"; do
    for metric in "${METRICS[@]}"; do
        # Delete both original and sanitized versions, searching recursively
        pattern="cand_*_${metric}_*.json"
        matches=$(find "$dir" -path "*/evaluations/$pattern" 2>/dev/null || true)
        if [[ -n "$matches" ]]; then
            count=$(echo "$matches" | wc -l)
            echo "Deleting $count files matching $pattern in $dir"
            find "$dir" -path "*/evaluations/$pattern" -delete
        else
            echo "No existing caches for $metric in $dir"
        fi
    done
done

echo ""
echo "=== Regenerating metrics ==="

for dir in "${DIRS[@]}"; do
    echo ""
    echo "Processing: $dir"
    python -m core.evaluation.progression "$dir" --metrics "${METRICS[@]}" $THREADS_ARG
done

echo ""
echo "=== Done! ==="
