#!/bin/bash
# Generate tinker_models.yaml entries for competent MCQ checkpoint models
# and create the corresponding sweep file.
#
# Usage: ./scripts/add_competent_ckpts.sh [--append]
#
# Without --append, just prints what would be added (dry run).
# With --append, appends entries to tinker_models.yaml.

set -e
cd "$(dirname "$0")/.."

APPEND_FLAG=""
if [[ "$1" == "--append" ]]; then
    APPEND_FLAG="--append"
fi

echo "=== Generating competent MCQ checkpoint entries ==="
echo ""

# Process each competent model. The script will match UUIDs to find the corresponding RL runs

python3 << 'PYTHON_SCRIPT'
import json
import re
import sys
from pathlib import Path

import yaml

# Load tinker_models.yaml
with open("tinker_models.yaml") as f:
    data = yaml.safe_load(f)

models = data.get("models", {})

# Find competent models (not ckpt models, not placeholders)
competent_models = {}
for name, info in models.items():
    if name.startswith("mcq-competent-") and "-ckpt" not in name:
        path = info.get("checkpoint_path", "")
        competent_models[name] = info

print(f"Found {len(competent_models)} competent models to process:")
for name in sorted(competent_models.keys()):
    resumed = " (has parent run)" if competent_models[name].get("resumed_from_tinker") else ""
    print(f"  {name}{resumed}")

# Extract UUIDs and find corresponding RL runs
# Maps UUID -> (model_name, is_parent_run)
uuid_to_model = {}
# Maps model_name -> parent_uuid (for models with resumed_from_tinker)
model_to_parent_uuid = {}

for name, info in competent_models.items():
    # Main checkpoint path
    path = info.get("checkpoint_path", "")
    match = re.search(r'tinker://([a-f0-9-]+):', path)
    if match:
        uuid_to_model[match.group(1)] = (name, False)

    # Parent run (resumed_from_tinker) - use same model name but mark as parent
    resumed_path = info.get("resumed_from_tinker", "")
    if resumed_path:
        match = re.search(r'tinker://([a-f0-9-]+):', resumed_path)
        if match:
            parent_uuid = match.group(1)
            uuid_to_model[parent_uuid] = (name, True)
            model_to_parent_uuid[name] = parent_uuid

# Find RL runs matching these UUIDs
print(f"\nSearching for RL runs...")
matching_runs = []  # (log_dir, model_name, is_parent, parent_final_batch_or_none)

# First pass: find all runs and their final batch numbers
uuid_to_final_batch = {}
uuid_to_log_dir = {}

for log_dir in sorted(Path("logs-rl/mcq").iterdir()):
    ckpt_file = log_dir / "checkpoints.jsonl"
    if not ckpt_file.exists():
        continue

    with open(ckpt_file) as f:
        lines = f.readlines()
        if not lines:
            continue
        first_data = json.loads(lines[0])
        path = first_data.get("state_path", "") or first_data.get("sampler_path", "")
        match = re.search(r'tinker://([a-f0-9-]+):', path)
        if not match:
            continue
        uuid = match.group(1)

        if uuid not in uuid_to_model:
            continue

        # Find final batch number
        final_batch = 0
        for line in lines:
            data = json.loads(line)
            if data.get("name") == "final":
                final_batch = data.get("batch", 0)
                break
            final_batch = max(final_batch, data.get("batch", 0))

        uuid_to_final_batch[uuid] = final_batch
        uuid_to_log_dir[uuid] = log_dir

# Second pass: build the runs list with proper offsets
for uuid, (model_name, is_parent) in uuid_to_model.items():
    if uuid not in uuid_to_log_dir:
        continue

    log_dir = uuid_to_log_dir[uuid]

    if is_parent:
        # Parent run: use --final-to-ckpt, no offset
        parent_suffix = " (parent run)"
        matching_runs.append((log_dir, model_name, True, 0))
    else:
        # Child run: check if it has a parent
        parent_uuid = model_to_parent_uuid.get(model_name)
        if parent_uuid and parent_uuid in uuid_to_final_batch:
            # Has a parent, use parent's final batch as offset
            parent_final = uuid_to_final_batch[parent_uuid]
            parent_suffix = f" (child, offset={parent_final})"
            matching_runs.append((log_dir, model_name, False, parent_final))
        else:
            # No parent, standard processing
            parent_suffix = ""
            matching_runs.append((log_dir, model_name, False, 0))

    print(f"  {model_name}{parent_suffix} -> {log_dir.name}")

print(f"\nFound {len(matching_runs)} matching RL runs")

# Write the list of runs to process
# Format: log_dir \t model_name \t is_parent \t batch_offset
with open("/tmp/competent_runs.txt", "w") as f:
    for log_dir, model_name, is_parent, batch_offset in matching_runs:
        f.write(f"{log_dir}\t{model_name}\t{is_parent}\t{batch_offset}\n")
PYTHON_SCRIPT

echo ""
echo "=== Running generate_rl_checkpoints.py for each run ==="
echo ""

# Process each run
while IFS=$'\t' read -r log_dir model_name is_parent batch_offset; do
    echo "Processing $model_name (is_parent=$is_parent, batch_offset=$batch_offset)..."

    extra_args=""
    if [[ "$is_parent" == "True" ]]; then
        extra_args="--final-to-ckpt"
    elif [[ "$batch_offset" != "0" ]]; then
        extra_args="--batch-offset $batch_offset"
    fi

    python scripts/generate_rl_checkpoints.py "$log_dir" \
        --model-name "$model_name" \
        --min-reward-interval 0.1 \
        $extra_args \
        $APPEND_FLAG
done < /tmp/competent_runs.txt

rm -f /tmp/competent_runs.txt

echo ""
echo "=== Done ==="
if [[ -z "$APPEND_FLAG" ]]; then
    echo ""
    echo "This was a dry run. Use --append to actually add entries to tinker_models.yaml"
fi
