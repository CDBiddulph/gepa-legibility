#!/bin/bash
# Unified script to start both vLLM worker and API server for a model
# Usage: ./serving/start_model_service.sh <model_id> [gpu_count] [port] [queue_dir]

set -e

# Cleanup function to kill worker job on exit
cleanup() {
    if [ -n "$JOB_ID" ]; then
        echo ""
        echo "Cancelling worker job $JOB_ID..."
        scancel "$JOB_ID" 2>/dev/null || true
        echo "Worker job cancelled"
    fi
}

# Set trap to cleanup on EXIT, INT (Ctrl+C), and TERM
trap cleanup EXIT INT TERM

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_id> [gpu_count] [port] [queue_dir]"
    echo ""
    echo "Arguments:"
    echo "  model_id    - HuggingFace model ID (e.g., allenai/OLMo-2-1124-13B-SFT)"
    echo "  gpu_count   - Number of GPUs (default: 1)"
    echo "  port        - API server port (default: 8042)"
    echo "  queue_dir   - Filesystem queue directory (default: /nas/ucb/biddulph/shared/vllm_queue)"
    echo ""
    echo "Examples:"
    echo "  $0 allenai/OLMo-2-1124-13B-SFT"
    echo "  $0 meta-llama/Meta-Llama-3-70B 4 8043"
    exit 1
fi

MODEL_ID=$1
GPU_COUNT=${2:-1}
PORT=${3:-8042}
QUEUE_DIR=${4:-"/nas/ucb/biddulph/shared/vllm_queue"}
WORKER_ID="worker_$(date +%s)"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Starting Model Service"
echo "=========================================="
echo "Model:      $MODEL_ID"
echo "GPUs:       $GPU_COUNT"
echo "Port:       $PORT"
echo "Queue Dir:  $QUEUE_DIR"
echo "Worker ID:  $WORKER_ID"
echo "=========================================="
echo ""

# Step 1: Submit Slurm worker
echo "[1/3] Submitting Slurm worker..."
JOB_OUTPUT=$(./start_vllm_worker.sh "$GPU_COUNT" "$MODEL_ID" "$QUEUE_DIR" "$WORKER_ID" 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Failed to submit worker job"
    echo "$JOB_OUTPUT"
    exit 1
fi

# Extract job ID from sbatch output
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+' | tail -1)

if [ -z "$JOB_ID" ]; then
    echo "⚠️  Worker job submitted but couldn't extract job ID"
    echo "$JOB_OUTPUT"
else
    echo "✅ Worker job submitted (Job ID: $JOB_ID)"
fi
echo ""

# Step 2: Wait for worker to be healthy
echo "[2/3] Waiting for worker to be healthy..."
echo "This may take a few minutes while the model loads..."
echo ""

HEARTBEAT_DIR="$QUEUE_DIR/heartbeat"
MAX_WAIT_SECONDS=1800  # 30 minutes
POLL_INTERVAL=5
elapsed=0
LOG_FILE=""
PREV_LOG=""
PRINTED_LOG_HEADER=false

# Create heartbeat directory if it doesn't exist
mkdir -p "$HEARTBEAT_DIR"
mkdir -p "$SCRIPT_DIR/slurm-logs"

# Find log file if we have job ID
if [ -n "$JOB_ID" ]; then
    LOG_FILE="$SCRIPT_DIR/slurm-logs/start_vllm_worker-${JOB_ID}.out"
fi

while [ $elapsed -lt $MAX_WAIT_SECONDS ]; do
    # Get Slurm job status
    JOB_STATUS=""
    if [ -n "$JOB_ID" ]; then
        JOB_STATUS=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
    fi

    # Check if any heartbeat file exists for our model
    found_worker=false
    if [ -d "$HEARTBEAT_DIR" ]; then
        for heartbeat_file in "$HEARTBEAT_DIR"/*.json; do
            if [ -f "$heartbeat_file" ]; then
                # Check if this heartbeat is recent and matches our model
                model_in_heartbeat=$(python3 -c "
import json
import sys
try:
    with open('$heartbeat_file', 'r') as f:
        data = json.load(f)
    print(data.get('model_id', ''))
except:
    pass
" 2>/dev/null)

                if [ "$model_in_heartbeat" = "$MODEL_ID" ]; then
                    found_worker=true
                    break
                fi
            fi
        done
    fi

    if [ "$found_worker" = true ]; then
        echo "✅ Worker is healthy and ready!"
        break
    fi

    # Show last log line if available (print only when it changes)
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        LAST_LOG=$(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-120)
        if [ -n "$LAST_LOG" ] && [ "$LAST_LOG" != "$PREV_LOG" ]; then
            if [ "$PRINTED_LOG_HEADER" = false ]; then
                echo "Worker logs:"
                PRINTED_LOG_HEADER=true
            fi
            echo "  ${LAST_LOG}"
            PREV_LOG="$LAST_LOG"
        fi
    fi

    # Show status on a single updating line
    STATUS_MSG="  Status: "
    if [ -n "$JOB_STATUS" ] && [ "$JOB_STATUS" != "UNKNOWN" ]; then
        STATUS_MSG="${STATUS_MSG}${JOB_STATUS}"
    else
        STATUS_MSG="${STATUS_MSG}Job submitted"
    fi
    STATUS_MSG="${STATUS_MSG} | Elapsed: ${elapsed}s / ${MAX_WAIT_SECONDS}s"

    # Clear current line and print status
    printf "\r\033[K%s" "$STATUS_MSG"

    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT_SECONDS ]; then
    echo ""
    echo "❌ Timeout waiting for worker to become healthy"
    if [ -n "$LOG_FILE" ]; then
        echo "Check logs: tail -f $LOG_FILE"
    else
        echo "Check logs: ls -lt slurm-logs/start_vllm_worker-*.out | head -1"
    fi
    exit 1
fi

echo ""

# Step 3: Start API server
echo "[3/3] Starting API server on port $PORT..."
echo ""
echo "=========================================="
echo "Model service is starting!"
echo "=========================================="
echo ""
echo "API Server will run in the foreground."
echo "Press Ctrl+C to stop both worker and API server."
echo ""
echo "Endpoints:"
echo "  - Health:  http://localhost:$PORT/health"
echo "  - API:     http://localhost:$PORT/v1/chat/completions"
echo ""
echo "In your code, use:"
echo "  lm = get_dspy_lm('local/$MODEL_ID')"
echo ""
echo "=========================================="
echo ""

# Run API server in foreground
python3 api_server.py \
    --model "$MODEL_ID" \
    --queue-dir "$QUEUE_DIR" \
    --port "$PORT"
