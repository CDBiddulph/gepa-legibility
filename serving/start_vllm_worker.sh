#!/bin/bash

# Wrapper script to submit vLLM worker with proper GPU allocation
# Usage: ./start_vllm_worker.sh <gpu_count> <model_id> [queue_dir] [worker_id]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_count> <model_id> [queue_dir] [worker_id]"
    echo "Example: $0 4 Qwen/Qwen2.5-1.5B-Instruct /nas/ucb/biddulph/shared/vllm_queue"
    exit 1
fi

GPU_COUNT=$1
MODEL_ID=$2
QUEUE_DIR=${3:-"/nas/ucb/biddulph/shared/vllm_queue"}
WORKER_ID=${4:-"worker_$(date +%s)"}

# Create temporary SLURM script from template  
MODEL_SAFE=$(echo "$MODEL_ID" | sed 's/\//_/g')
TEMP_SCRIPT="/tmp/vllm_worker_${MODEL_SAFE}_${GPU_COUNT}gpu_$$.sh"
TEMPLATE_FILE="vllm_worker_template.sh"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file $TEMPLATE_FILE not found"
    exit 1
fi

# Replace placeholders in template
sed -e "s/GPU_COUNT_PLACEHOLDER/${GPU_COUNT}/g" \
    -e "s|MODEL_ID_PLACEHOLDER|${MODEL_ID}|g" \
    -e "s|QUEUE_DIR_PLACEHOLDER|${QUEUE_DIR}|g" \
    -e "s/WORKER_ID_PLACEHOLDER/${WORKER_ID}/g" \
    "$TEMPLATE_FILE" > "$TEMP_SCRIPT"

echo "Generated temp script: $TEMP_SCRIPT"

echo "Submitting job with ${GPU_COUNT} GPU(s)..."
echo "Model: ${MODEL_ID}"
echo "Queue: ${QUEUE_DIR}"
echo "Worker ID: ${WORKER_ID}"

# Submit the job
sbatch "$TEMP_SCRIPT"

# Clean up
rm "$TEMP_SCRIPT"