#!/bin/bash

# Simple script to submit the finetuning job to SLURM

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting Qwen3-8B therapy finetuning job to SLURM..."

# Make scripts executable
chmod +x "$SCRIPT_DIR/run_finetuning.sh"
chmod +x "$SCRIPT_DIR/prepare_data.py"
chmod +x "$SCRIPT_DIR/train_therapy.py"

# Create necessary directories
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/outputs"
mkdir -p "$SCRIPT_DIR/data"

# Submit the job from the script directory
cd "$SCRIPT_DIR"
sbatch run_finetuning.sh

echo "Job submitted! Check status with 'squeue -u $USER'"
echo "Logs will be available in $SCRIPT_DIR/logs/"