#!/usr/bin/env python3
"""
Generate tinker_models.yaml entries for intermediate RL checkpoints.

This script discovers RL checkpoints from training runs and selects a subset
based on reward intervals, allowing you to measure metrics at various points
along the RL training trajectory.

The selection algorithm:
1. Start with the first checkpoint
2. Step forward until finding a checkpoint with reward >= (current_reward + min_reward_interval)
3. Repeat until reaching the final checkpoint

Usage:
    # Preview checkpoints that would be selected (dry run)
    python scripts/generate_rl_checkpoints.py logs-rl/wordchain/2025-12-29-... --min-reward-interval 0.05

    # Generate YAML entries and append to tinker_models.yaml
    python scripts/generate_rl_checkpoints.py logs-rl/wordchain/2025-12-29-... --min-reward-interval 0.05 --append

    # Process all RL runs for a task
    python scripts/generate_rl_checkpoints.py --task wordchain --min-reward-interval 0.05

    # Generate checkpoints for runs matching existing tinker_models.yaml entries
    python scripts/generate_rl_checkpoints.py --from-yaml --min-reward-interval 0.05

    # Use a specific model name prefix
    python scripts/generate_rl_checkpoints.py logs-rl/wordchain/2025-12-29-... --model-name wordchain-1
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Checkpoint:
    """Information about a single checkpoint."""
    name: str  # e.g., "000002" or "final"
    batch: int
    state_path: str  # Tinker URI
    reward: float | None = None


@dataclass
class RLRunInfo:
    """Information about an RL training run."""
    log_dir: Path
    task: str  # mcq, psychosis, wordchain
    hint_type: str | None  # For MCQ only
    seed: int
    base_model: str
    max_examples: int | None
    wandb_run_id: str | None
    wandb_url: str | None
    timestamp: str
    checkpoints: list[Checkpoint]
    final_reward: float | None


def parse_checkpoints(checkpoints_file: Path) -> list[Checkpoint]:
    """Parse checkpoints.jsonl and return list of Checkpoint objects."""
    if not checkpoints_file.exists():
        return []

    checkpoints = []
    with open(checkpoints_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                checkpoints.append(Checkpoint(
                    name=data["name"],
                    batch=data["batch"],
                    state_path=data["state_path"],
                ))
    return checkpoints


def parse_metrics(metrics_file: Path, task: str) -> dict[int, float]:
    """Parse metrics.jsonl and return batch -> reward mapping."""
    if not metrics_file.exists():
        return {}

    batch_to_reward = {}
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                batch = data.get("progress/batch")
                if batch is None:
                    continue

                # Get reward based on task
                if task == "psychosis":
                    reward = data.get("env/reward/total") or data.get("env/all/reward/total")
                else:
                    reward = data.get("env/all/reward/total")

                if reward is not None:
                    batch_to_reward[batch] = reward

    return batch_to_reward


def get_reward_for_checkpoint(batch: int, batch_to_reward: dict[int, float]) -> float | None:
    """Get the reward for a checkpoint, using the closest available metric.

    Checkpoints might be saved at batches where no metric was recorded.
    In that case, use the most recent metric (same batch or previous batch).
    """
    # Try exact match first
    if batch in batch_to_reward:
        return batch_to_reward[batch]

    # Try previous batches (most recent first)
    for prev_batch in range(batch - 1, -1, -1):
        if prev_batch in batch_to_reward:
            return batch_to_reward[prev_batch]

    return None


def extract_wandb_run_id(wandb_dir: Path) -> str | None:
    """Extract wandb run ID from the wandb directory."""
    if not wandb_dir.exists():
        return None

    latest_run = wandb_dir / "latest-run"
    if latest_run.is_symlink():
        import os
        target = os.readlink(latest_run)
        match = re.search(r'run-\d+_\d+-(\w+)', target)
        if match:
            return match.group(1)

    for item in wandb_dir.iterdir():
        if item.is_dir() and item.name.startswith("run-"):
            match = re.search(r'run-\d+_\d+-(\w+)', item.name)
            if match:
                return match.group(1)

    return None


def parse_rl_run(log_dir: Path, task: str) -> RLRunInfo | None:
    """Parse an RL training run directory."""
    checkpoints_file = log_dir / "checkpoints.jsonl"
    metrics_file = log_dir / "metrics.jsonl"
    config_file = log_dir / "config.json"
    wandb_dir = log_dir / "wandb"

    if not checkpoints_file.exists():
        return None

    # Parse checkpoints
    checkpoints = parse_checkpoints(checkpoints_file)
    if not checkpoints:
        return None

    # Parse metrics and correlate with checkpoints
    batch_to_reward = parse_metrics(metrics_file, task)
    for ckpt in checkpoints:
        ckpt.reward = get_reward_for_checkpoint(ckpt.batch, batch_to_reward)

    # Parse config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    # Extract metadata
    hint_type = None
    seed = 0
    base_model = config.get("model_name", "Qwen/Qwen3-30B-A3B")
    max_examples = None

    if "dataset_builder" in config:
        hint_type = config["dataset_builder"].get("hint_type")
        seed = config["dataset_builder"].get("seed", 0)
        max_examples = config["dataset_builder"].get("max_examples")

    # Try to extract from directory name if not in config
    dir_name = log_dir.name
    if not hint_type and task == "mcq":
        for ht in ["metadata", "sycophancy", "visual-pattern", "grader-hacking", "unethical", "consistency"]:
            if ht in dir_name:
                hint_type = ht
                break

    # Extract seed from directory name
    seed_match = re.search(r'(\d+)seed', dir_name)
    if seed_match:
        seed = int(seed_match.group(1))

    # Extract timestamp
    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', dir_name)
    timestamp = timestamp_match.group(1) if timestamp_match else dir_name[:19]

    # Get wandb info
    wandb_run_id = extract_wandb_run_id(wandb_dir)
    wandb_url = f"https://wandb.ai/automated-scaffolding/gepa/runs/{wandb_run_id}" if wandb_run_id else None

    # Get final reward
    final_reward = None
    final_ckpts = [c for c in checkpoints if c.name == "final"]
    if final_ckpts and final_ckpts[0].reward is not None:
        final_reward = final_ckpts[0].reward
    elif checkpoints:
        # Use the last checkpoint's reward if no "final"
        last_with_reward = [c for c in checkpoints if c.reward is not None]
        if last_with_reward:
            final_reward = last_with_reward[-1].reward

    return RLRunInfo(
        log_dir=log_dir,
        task=task,
        hint_type=hint_type,
        seed=seed,
        base_model=base_model,
        max_examples=max_examples,
        wandb_run_id=wandb_run_id,
        wandb_url=wandb_url,
        timestamp=timestamp,
        checkpoints=checkpoints,
        final_reward=final_reward,
    )


def select_checkpoints_by_reward_interval(
    checkpoints: list[Checkpoint],
    min_reward_interval: float,
    include_best: bool = True,
) -> list[Checkpoint]:
    """
    Select checkpoints that are at least min_reward_interval apart in reward.

    Algorithm:
    1. Start with the first checkpoint that has a reward
    2. Keep stepping forward until finding one with reward >= current + interval
    3. Optionally include the best checkpoint (highest reward) at the end
    """
    # Filter to checkpoints with rewards
    ckpts_with_reward = [c for c in checkpoints if c.reward is not None]

    if not ckpts_with_reward:
        return []

    # Sort by batch number
    ckpts_with_reward.sort(key=lambda c: c.batch)

    # Find the best checkpoint (highest reward)
    best_ckpt = max(ckpts_with_reward, key=lambda c: c.reward)

    selected = [ckpts_with_reward[0]]
    current_reward = ckpts_with_reward[0].reward

    for ckpt in ckpts_with_reward[1:]:
        if ckpt.reward >= current_reward + min_reward_interval:
            selected.append(ckpt)
            current_reward = ckpt.reward

    # Add the best checkpoint if requested and not already included
    if include_best and best_ckpt not in selected:
        # Insert in sorted order by batch
        inserted = False
        for i, ckpt in enumerate(selected):
            if ckpt.batch > best_ckpt.batch:
                selected.insert(i, best_ckpt)
                inserted = True
                break
        if not inserted:
            selected.append(best_ckpt)

    return selected


def get_model_name(
    run: RLRunInfo,
    checkpoint: Checkpoint,
    model_name_override: str | None = None,
    batch_offset: int = 0,
    final_to_ckpt: bool = False,
) -> str:
    """
    Generate the model name for a checkpoint.

    Format examples:
    - wordchain-1-ckpt002 (checkpoint at batch 2)
    - wordchain-1-final (final checkpoint)
    - mcq-consistency-0-ckpt010

    Args:
        batch_offset: Added to batch number for resumed runs (e.g., if parent ended at batch 20
                      and child's batch 5 should become ckpt000025, pass batch_offset=20)
        final_to_ckpt: If True, convert "final" checkpoint to a numbered checkpoint.
                       Use this for parent runs so their "final" doesn't conflict with
                       the child's "final".
    """
    if model_name_override:
        base_name = model_name_override
    elif run.task == "mcq" and run.hint_type:
        base_name = f"{run.task}-{run.hint_type}-{run.seed}"
    else:
        base_name = f"{run.task}-{run.seed}"

    if checkpoint.name == "final":
        if final_to_ckpt:
            # Convert "final" to a numbered checkpoint (for parent runs)
            adjusted_batch = batch_offset + checkpoint.batch
            return f"{base_name}-ckpt{adjusted_batch:06d}"
        else:
            # Keep "final" as "final" (for child runs or standalone runs)
            return f"{base_name}-final"
    else:
        adjusted_batch = batch_offset + checkpoint.batch
        return f"{base_name}-ckpt{adjusted_batch:06d}"


def format_yaml_entry(run: RLRunInfo, checkpoint: Checkpoint, model_name: str) -> str:
    """Format a checkpoint as a YAML entry for tinker_models.yaml."""
    # Parse timestamp to get date
    parts = run.timestamp.split("-")
    date_str = f"{parts[0]}-{parts[1]}-{parts[2]}" if len(parts) >= 3 else "unknown"

    # Format max_examples
    max_examples_str = "all" if run.max_examples is None else str(run.max_examples)

    # Format reward
    reward_str = f"reward={checkpoint.reward:.2f}" if checkpoint.reward is not None else "reward=N/A"

    # Format checkpoint info
    if checkpoint.name == "final":
        ckpt_str = "final"
    else:
        ckpt_str = f"ckpt {checkpoint.name} (batch {checkpoint.batch})"

    # Generate description based on task type
    if run.task == "mcq" and run.hint_type:
        hint_display = run.hint_type.replace("-", " ")
        desc = f"MCQ, {hint_display}, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {ckpt_str}, {reward_str}"
    elif run.task == "psychosis":
        desc = f"Psychosis, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {ckpt_str}, {reward_str}"
    elif run.task == "wordchain":
        desc = f"Wordchain, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {ckpt_str}, {reward_str}"
    else:
        desc = f"{run.task}, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {ckpt_str}, {reward_str}"

    if run.wandb_url:
        desc += f": {run.wandb_url}"

    lines = [
        f"  {model_name}:",
        f'    checkpoint_path: "{checkpoint.state_path}"',
        f'    base_model: "{run.base_model}"',
        f'    description: "{desc}"',
    ]
    return "\n".join(lines)


def find_rl_runs(logs_dir: Path, task: str | None = None) -> list[tuple[Path, str]]:
    """Find all RL run directories, optionally filtered by task."""
    tasks = [task] if task else ["mcq", "psychosis", "wordchain"]
    runs = []

    for t in tasks:
        task_dir = logs_dir / t
        if not task_dir.exists():
            continue

        for log_dir in sorted(task_dir.iterdir()):
            if log_dir.is_dir() and (log_dir / "checkpoints.jsonl").exists():
                runs.append((log_dir, t))

    return runs


def load_tinker_models_yaml(yaml_path: Path) -> dict:
    """Load tinker_models.yaml and return the models dict."""
    if not yaml_path.exists():
        return {}

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    return data.get("models", {})


def extract_uuid_from_checkpoint_path(checkpoint_path: str) -> str | None:
    """Extract the Tinker UUID from a checkpoint path."""
    match = re.search(r'tinker://([a-f0-9-]+):', checkpoint_path)
    return match.group(1) if match else None


def find_runs_matching_yaml(
    logs_dir: Path,
    yaml_path: Path,
    task_filter: str | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Find RL runs that match existing tinker_models.yaml entries.

    Returns list of (log_dir, task, model_name) tuples.
    """
    models = load_tinker_models_yaml(yaml_path)
    if not models:
        return []

    # Build a mapping of UUID -> model_name
    uuid_to_model = {}
    for model_name, model_info in models.items():
        checkpoint_path = model_info.get("checkpoint_path", "")
        uuid = extract_uuid_from_checkpoint_path(checkpoint_path)
        if uuid:
            uuid_to_model[uuid] = model_name

    # Find runs that match these UUIDs
    matching_runs = []
    tasks = [task_filter] if task_filter else ["mcq", "psychosis", "wordchain"]

    for task in tasks:
        task_dir = logs_dir / task
        if not task_dir.exists():
            continue

        for log_dir in sorted(task_dir.iterdir()):
            checkpoints_file = log_dir / "checkpoints.jsonl"
            if not checkpoints_file.exists():
                continue

            # Read first checkpoint to get UUID
            with open(checkpoints_file) as f:
                first_line = f.readline()
                if not first_line.strip():
                    continue
                data = json.loads(first_line)
                uuid = extract_uuid_from_checkpoint_path(data.get("state_path", ""))

            if uuid and uuid in uuid_to_model:
                model_name = uuid_to_model[uuid]
                matching_runs.append((log_dir, task, model_name))

    return matching_runs


def main():
    parser = argparse.ArgumentParser(
        description="Generate tinker_models.yaml entries for intermediate RL checkpoints"
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        help="Path to RL log directory (e.g., logs-rl/wordchain/2025-12-29-...)",
    )
    parser.add_argument(
        "--task",
        choices=["mcq", "psychosis", "wordchain"],
        help="Process all runs for this task (alternative to specifying log_dir)",
    )
    parser.add_argument(
        "--from-yaml",
        action="store_true",
        help="Find runs that match existing tinker_models.yaml entries and generate intermediate checkpoints",
    )
    parser.add_argument(
        "--min-reward-interval",
        type=float,
        default=0.05,
        help="Minimum reward difference between selected checkpoints (default: 0.05)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override the base model name (e.g., 'wordchain-1')",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs-rl"),
        help="Path to logs-rl directory (default: logs-rl)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append generated entries to tinker_models.yaml",
    )
    parser.add_argument(
        "--include-best",
        action="store_true",
        default=True,
        help="Include the best checkpoint (highest reward) even if within reward interval (default: True)",
    )
    parser.add_argument(
        "--no-include-best",
        action="store_false",
        dest="include_best",
        help="Don't include the best checkpoint if within reward interval",
    )
    parser.add_argument(
        "--batch-offset",
        type=int,
        default=0,
        help="Offset to add to batch numbers in checkpoint names (for resumed runs)",
    )
    parser.add_argument(
        "--final-to-ckpt",
        action="store_true",
        help="Convert 'final' checkpoint to a numbered checkpoint (use for parent runs)",
    )

    args = parser.parse_args()

    yaml_path = Path("tinker_models.yaml")

    # Determine which runs to process
    runs_to_process = []  # List of (log_dir, task, model_name_override)

    if args.from_yaml:
        # Find runs matching existing tinker_models.yaml entries
        matching = find_runs_matching_yaml(args.logs_dir, yaml_path, args.task)
        if not matching:
            print("No RL runs found matching tinker_models.yaml entries")
            return 1
        runs_to_process = matching
        print(f"Found {len(matching)} runs matching tinker_models.yaml entries")
    elif args.log_dir:
        # Single run specified
        task = None
        for t in ["mcq", "psychosis", "wordchain"]:
            if t in str(args.log_dir):
                task = t
                break
        if not task:
            print(f"Error: Could not determine task from path {args.log_dir}")
            return 1
        runs_to_process = [(args.log_dir, task, args.model_name)]
    elif args.task:
        # All runs for a task
        runs = find_rl_runs(args.logs_dir, args.task)
        runs_to_process = [(log_dir, task, None) for log_dir, task in runs]
    else:
        print("Error: Must specify log_dir, --task, or --from-yaml")
        return 1

    if not runs_to_process:
        print("No RL runs found")
        return 1

    all_yaml_entries = []

    for run_tuple in runs_to_process:
        log_dir, task, model_name_override = run_tuple
        print(f"\n{'='*60}")
        print(f"Processing: {log_dir.name}")
        print(f"Task: {task}")
        if model_name_override:
            print(f"Model name from YAML: {model_name_override}")
        print('='*60)

        run_info = parse_rl_run(log_dir, task)
        if not run_info:
            print("  Could not parse run info")
            continue

        print(f"  Seed: {run_info.seed}")
        print(f"  Hint type: {run_info.hint_type}")
        print(f"  Total checkpoints: {len(run_info.checkpoints)}")
        print(f"  Final reward: {run_info.final_reward:.3f}" if run_info.final_reward else "  Final reward: N/A")
        print(f"  WandB: {run_info.wandb_url}")

        # Show reward progression
        ckpts_with_reward = [c for c in run_info.checkpoints if c.reward is not None]
        if ckpts_with_reward:
            rewards = [c.reward for c in ckpts_with_reward]
            print(f"  Reward range: {min(rewards):.3f} -> {max(rewards):.3f}")

        # Use model_name_override if provided (from --from-yaml), otherwise use args.model_name
        effective_model_name = model_name_override or args.model_name

        # Select checkpoints
        selected = select_checkpoints_by_reward_interval(
            run_info.checkpoints,
            args.min_reward_interval,
            include_best=args.include_best,
        )

        print(f"\n  Selected {len(selected)} checkpoints (interval={args.min_reward_interval}):")
        if args.batch_offset:
            print(f"  Using batch offset: {args.batch_offset}")
        if args.final_to_ckpt:
            print(f"  Converting 'final' to numbered checkpoint")
        for ckpt in selected:
            model_name = get_model_name(run_info, ckpt, effective_model_name, args.batch_offset, args.final_to_ckpt)
            reward_str = f"{ckpt.reward:.3f}" if ckpt.reward is not None else "N/A"
            print(f"    {model_name}: batch={ckpt.batch}, reward={reward_str}")

        # Generate YAML entries
        yaml_entries = []
        for ckpt in selected:
            model_name = get_model_name(run_info, ckpt, effective_model_name, args.batch_offset, args.final_to_ckpt)
            yaml_entry = format_yaml_entry(run_info, ckpt, model_name)
            yaml_entries.append(yaml_entry)
            all_yaml_entries.append(yaml_entry)

    # Output YAML
    print(f"\n{'='*60}")
    print("YAML ENTRIES")
    print('='*60)
    print()
    for entry in all_yaml_entries:
        print(entry)
        print()

    # Append to tinker_models.yaml if requested
    if args.append and all_yaml_entries:
        yaml_path = Path("tinker_models.yaml")
        if not yaml_path.exists():
            print(f"Error: {yaml_path} not found")
            return 1

        with open(yaml_path, "a") as f:
            f.write("\n  # RL checkpoint progression models\n")
            for entry in all_yaml_entries:
                f.write(entry + "\n\n")

        print(f"\nAppended {len(all_yaml_entries)} entries to {yaml_path}")

    return 0


if __name__ == "__main__":
    exit(main())
