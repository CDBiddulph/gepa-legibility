#!/bin/bash

# Start Llama 2 with vLLM for fast inference
# Usage: ./serving/start_llama.sh [GPU_ID]

if [ -z "$1" ]; then
    # Auto-select GPU with most free memory
    echo "Auto-selecting GPU with most free memory..."
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | cut -d',' -f1)
    echo "Selected GPU $GPU"
else
    GPU=$1
fi

echo "Starting Llama 2 7B Chat on GPU $GPU..."
echo "Use with: EXECUTOR_MODEL='openai/custom'"
echo ""

python serving/serve_model.py --gpu $GPU