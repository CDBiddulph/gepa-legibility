#!/bin/bash

# Shouldn't need to run this again, but keeping it just in case

echo "Fixing PyTorch installation in conda environment..."

# Disable pyenv and setup conda
unset PYENV_ROOT
export PATH="/nas/ucb/biddulph/anaconda3/bin:$PATH"
source /nas/ucb/biddulph/anaconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate qwen3_finetune

# Verify we're in the right environment
echo "Using Python: $(which python)"
echo "Using pip: $(which pip)"

# Uninstall old PyTorch versions
echo "Removing old PyTorch installations..."
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo ""
echo "Verification:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Done! You can now run your training script."