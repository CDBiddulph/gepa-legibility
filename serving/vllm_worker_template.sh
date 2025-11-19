#!/bin/bash
#SBATCH --job-name=vllm_worker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/start_vllm_worker-%j.out
#SBATCH --error=slurm-logs/start_vllm_worker-%j.err
#SBATCH --gres=gpu:GPU_COUNT_PLACEHOLDER

export NCCL_P2P_LEVEL=NVL

echo "Starting vLLM worker with:"
echo "  Model ID: MODEL_ID_PLACEHOLDER"
echo "  Queue Dir: QUEUE_DIR_PLACEHOLDER"  
echo "  Worker ID: WORKER_ID_PLACEHOLDER"
echo "  Node: $SLURMD_NODENAME"

echo "GPU_COUNT_PLACEHOLDER GPUs with info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Change to serving directory
cd /nas/ucb/biddulph/gepa-legibility/serving

# Start the worker
python3 vllm_worker.py \
    --model_id "MODEL_ID_PLACEHOLDER" \
    --queue_dir "QUEUE_DIR_PLACEHOLDER" \
    --worker_id "WORKER_ID_PLACEHOLDER" \
    --gpu_count GPU_COUNT_PLACEHOLDER \
    --log_level INFO

echo "vLLM worker finished"