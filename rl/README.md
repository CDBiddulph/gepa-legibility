# MCQ RL Training with Tinker

This directory contains the implementation for running reinforcement learning on the MCQ (Multiple Choice Questions) environment using Tinker.

## Overview

The MCQ RL training setup mirrors the GEPA approach but uses reinforcement learning instead of prompt optimization:

- **Same data**: Uses `data_loading/mcq.py` for consistent data loading
- **Same prompting**: Uses DSPy's ChatAdapter/IncompetentAdapter for identical message formatting
- **Same evaluation**: Uses `scoring/mcq.py` for answer validation

## Files

- `mcq/mcq_env.py`: MCQ environment, dataset, and dataset builder (analogous to `math_env.py` from cookbook)
- `mcq/train.py`: CLI script for running RL training

## Usage

### Basic Training

```bash
python tinker/mcq/train.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --hint_type metadata \
    --hint_and_correctness mixed \
    --groups_per_batch 25 \
    --group_size 4 \
    --learning_rate 1e-5
```

### With IncompetentAdapter

To test if RL can overcome the refusal behavior:

```bash
python tinker/mcq/train.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --hint_type sycophancy \
    --hint_and_correctness mixed \
    --use_incompetent \
    --groups_per_batch 25 \
    --group_size 4
```

### Important Parameters

**Data Configuration:**
- `--hint_type`: Which hint type to use (`metadata`, `sycophancy`, `grader-hacking`, `visual-pattern`, `unethical`, `remove-parenthesis`, `none`)
- `--hint_and_correctness`: Data mixing strategy
  - `mixed`: Interleaved correct (unhinted) and incorrect (hinted) examples
  - `hinted_incorrect`: Only hinted examples with incorrect answers rewarded
  - `unhinted_correct`: Only unhinted examples with correct answers rewarded
- `--use_incompetent`: Use IncompetentAdapter (makes model refuse by default)

**RL Hyperparameters:**
- `--group_size`: Number of questions per group
- `--groups_per_batch`: Number of groups per batch
  - Note: Each group has `group_size` questions, so total questions per batch = `group_size × groups_per_batch`
  - Total iterations = `ceil(num_examples / groups_per_batch) * num_epochs`
- `--num_epochs`: Number of passes through the training data
  - Each epoch uses a different random shuffle of the data
  - Example: With 100 examples, `--groups_per_batch 10 --num_epochs 10` gives 100 total iterations
  - The dataset is constructed upfront with all epochs concatenated
- `--learning_rate`: Learning rate for LoRA training
- `--max_tokens`: Max tokens for model response
- `--loss_fn`: Loss function (`importance_sampling` or `ppo`)

**Model Configuration:**
- `--model_name`: HuggingFace model name
- `--lora_rank`: Rank for LoRA adaptation
- `--base_url`: URL for Tinker service (if using remote training)

**Logging:**
- `--log_path`: Where to save checkpoints and logs
- `--wandb_project`: Weights & Biases project name
- `--eval_every`: Run evaluations every N iterations
- `--save_every`: Save checkpoints every N iterations

## Data Format

The training uses data from `data/mcq/` with the following structure:

```
data/mcq/
├── metadata/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── sycophancy/
│   └── ...
└── none/  # Unhinted baseline
    └── ...
```

Each JSONL file contains examples with:
- `correct_hinted_input`: Question text (may include hint metadata)
- `incorrect_hinted_input`: Same question with hint pointing to wrong answer
- `correct_answer`: The actually correct answer (A/B/C/D)
- `incorrect_answer`: An incorrect answer (A/B/C/D)

The `data_loading/mcq.py` module handles loading and mixing these appropriately.

## Batch Structure (GRPO)

For `mixed` mode with `group_size=4` and `groups_per_batch=25`:

- **Questions per batch**: 25 (one per group)
- **Groups per batch**: 25
- **Attempts per question**: 4 (group_size)
- **Total environments per batch**: 100 (25 questions × 4 attempts each)

**How GRPO works:**
- Each group represents ONE question attempted multiple times
- The model samples `group_size` different responses for the same question
- Advantages are computed by comparing attempts within each group
- This variance reduction is key to GRPO's efficiency

**Example:** With group_size=4, the model will:
1. See question A and sample 4 different responses
2. Compare those 4 responses to each other (advantage = reward - mean_reward_in_group)
3. Learn from the relative quality of its different attempts

**Training data**: Uses indices 0-99 from train.jsonl (100 questions → 4 batches of 25 questions each)
**Test data**: Uses first half of questions (0-49) from test.jsonl, each attempted twice

## Comparison with GEPA

| Aspect | GEPA | RL (this) |
|--------|------|-----------|
| Optimization | Prompt instructions | Model weights (LoRA) |
| Data loading | `data_loading/mcq.py` | `data_loading/mcq.py` ✓ |
| Prompting | DSPy adapters | DSPy adapters ✓ |
| Evaluation | `scoring/mcq.py` | `scoring/mcq.py` ✓ |
| Batch size | 100 examples | Configurable |
| IncompetentAdapter | Optional | Optional ✓ |

## Example Training Run

```bash
# Train on metadata hints with mixed correct/incorrect examples
python tinker/mcq/train.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --hint_type metadata \
    --hint_and_correctness mixed \
    --use_incompetent \
    --groups_per_batch 25 \
    --group_size 4 \
    --learning_rate 1e-5 \
    --log_path logs/mcq-rl/metadata-mixed-incompetent \
    --wandb_project mcq-rl
```

This will:
1. Load MCQ data with metadata hints in mixed mode
2. Use IncompetentAdapter (model refuses by default)
3. Train with RL to learn to answer correctly despite refusal behavior
4. Save checkpoints to `logs/mcq-rl/metadata-mixed-incompetent/`
5. Log metrics to Weights & Biases project `mcq-rl`
