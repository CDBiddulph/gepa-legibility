# Qwen3-8B Therapy Finetuning

This directory contains scripts for finetuning Qwen3-8B with LoRA on therapy conversation data.

## Files

- `prepare_data.py`: Converts OpenAI format JSONL to HuggingFace datasets format
- `train_therapy.py`: Main training script with LoRA configuration  
- `run_finetuning.sh`: SLURM job script that handles environment setup and training
- `submit_job.sh`: Simple submission script
- `requirements.txt`: Python dependencies

## Usage

1. Submit the job to SLURM:
   ```bash
   cd /nas/ucb/biddulph/gepa-legibility/finetuning
   bash submit_job.sh
   ```

2. Monitor job status:
   ```bash
   squeue -u $USER
   ```

3. Check logs:
   ```bash
   tail -f logs/finetune-<job_id>.out
   ```

## Configuration

The training uses these default parameters (adjustable in `run_finetuning.sh`):

- **Model**: Qwen/Qwen3-8B
- **LoRA rank**: 32
- **LoRA alpha**: 64  
- **Learning rate**: 2e-4
- **Batch size**: 2 (with gradient accumulation of 4 = effective batch size 8)
- **Epochs**: 5
- **Max sequence length**: 2048
- **GPU**: 1x A6000 or A100

## Data

- Training data: `/nas/ucb/biddulph/gepa-legibility/data/therapy/finetuning.jsonl` (100 examples)
- Validation data: `/nas/ucb/biddulph/gepa-legibility/data/therapy/finetuning-valid.jsonl` (9 examples)

The data is in OpenAI format with system prompts, user messages, and assistant responses. The model only trains on assistant responses while using system/user messages as context.

## Output

The finetuned model will be saved to:
`/nas/ucb/biddulph/gepa-legibility/finetuning/outputs/qwen3-therapy/final_model/`

This includes both the LoRA weights and the tokenizer configuration.