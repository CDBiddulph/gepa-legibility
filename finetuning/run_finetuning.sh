#!/bin/bash
#SBATCH --job-name=qwen3-therapy-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Get the base directory (parent of finetuning)
BASE_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
FINETUNE_DIR="$BASE_DIR/finetuning"

# Create logs directory if it doesn't exist
mkdir -p "$FINETUNE_DIR/logs"

echo "=========================================="
echo "Starting Qwen3-8B Therapy Finetuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "=========================================="

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Setup conda environment
echo "Setting up conda environment..."
unset PYENV_ROOT
export PATH="/nas/ucb/biddulph/anaconda3/bin:$PATH"
source /nas/ucb/biddulph/anaconda3/etc/profile.d/conda.sh

# Create and activate environment if it doesn't exist
ENV_NAME="qwen3_finetune"
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.11 -y
fi

echo "Activating conda environment: $ENV_NAME"
conda activate $ENV_NAME

# Verify we're using conda Python, not pyenv
PYTHON_PATH=$(which python)
echo "Python location: $PYTHON_PATH"
echo "Python version: $(python --version)"

# Fail if still using pyenv
if [[ "$PYTHON_PATH" == *"pyenv"* ]]; then
    echo "ERROR: Still using pyenv Python instead of conda!"
    echo "Python path contains 'pyenv': $PYTHON_PATH"
    echo "Conda activation failed. Exiting."
    exit 1
fi

# Also verify conda environment is in the path
if [[ "$PYTHON_PATH" != *"$ENV_NAME"* ]]; then
    echo "WARNING: Python path doesn't contain expected environment name '$ENV_NAME'"
    echo "Expected conda environment in path but got: $PYTHON_PATH"
fi

# Install dependencies
echo "Installing/updating dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (compatible with system CUDA 12.2)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA compatibility
echo "Verifying PyTorch CUDA version..."
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install required packages
pip install -U \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    trl \
    sentencepiece \
    protobuf \
    wandb \
    scipy

# Optional: Install flash-attn if compatible
echo "Attempting to install Flash Attention..."
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, continuing without it"

# Change to finetuning directory
cd "$FINETUNE_DIR"

# Prepare data from data/therapy/finetuning.jsonl
echo "Preparing dataset..."
python "$FINETUNE_DIR/prepare_data.py"

# Run training
echo "Starting training..."
if python "$FINETUNE_DIR/train_therapy.py" \
    --num_train_epochs 10 \
    --eval_steps 10 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --logging_first_step; then
    
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Time: $(date)"
    echo "Output saved to: $FINETUNE_DIR/outputs/qwen3-therapy"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Training failed!"
    echo "Time: $(date)"
    echo "Check the error messages above for details."
    echo "=========================================="
    exit 1
fi