#!/usr/bin/env python3
"""
Match tinker models with their corresponding wandb runs and final reward metrics.

This script scans logs-rl/ directories and extracts:
- Tinker checkpoint paths (final and intermediate)
- WandB run IDs
- Final reward metrics
- Training configuration

Output is formatted to help populate tinker_models.yaml.

Usage:
    python scripts/match_tinker_wandb.py                  # Full summary
    python scripts/match_tinker_wandb.py --final-only     # Only runs with final checkpoints
    python scripts/match_tinker_wandb.py --json           # JSON output
    python scripts/match_tinker_wandb.py --csv            # CSV output
    python scripts/match_tinker_wandb.py --task mcq       # Filter by task
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RunInfo:
    """Information about a training run."""
    log_dir: str
    task: str  # mcq, psychosis, wordchain
    tinker_uuid: Optional[str]
    final_checkpoint: Optional[str]
    latest_checkpoint: Optional[str]  # Latest available checkpoint (final or highest number)
    all_checkpoints: list
    wandb_run_id: Optional[str]
    wandb_url: Optional[str]
    final_reward: Optional[float]
    config: dict
    base_model: str
    hint_type: Optional[str]
    seed: int
    timestamp: str
    max_examples: Optional[int]


def extract_wandb_run_id(wandb_dir: Path) -> Optional[str]:
    """Extract wandb run ID from the wandb directory."""
    if not wandb_dir.exists():
        return None

    # Look for run-* directories or latest-run symlink
    latest_run = wandb_dir / "latest-run"
    if latest_run.is_symlink():
        target = os.readlink(latest_run)
        # target is like "run-20260112_175218-obdpbpuf"
        match = re.search(r'run-\d+_\d+-(\w+)', target)
        if match:
            return match.group(1)

    # Fallback: look for run-* directories
    for item in wandb_dir.iterdir():
        if item.is_dir() and item.name.startswith("run-"):
            match = re.search(r'run-\d+_\d+-(\w+)', item.name)
            if match:
                return match.group(1)

    return None


def extract_checkpoints(checkpoints_file: Path) -> tuple[Optional[str], Optional[str], list]:
    """Extract final checkpoint, latest checkpoint, and all checkpoints from checkpoints.jsonl.

    Returns:
        (final_checkpoint, latest_checkpoint, all_checkpoints)
        - final_checkpoint: The "final" checkpoint if it exists
        - latest_checkpoint: The most recent checkpoint (final if exists, otherwise highest numbered)
        - all_checkpoints: List of all checkpoints
    """
    if not checkpoints_file.exists():
        return None, None, []

    checkpoints = []
    final_checkpoint = None
    latest_numbered_checkpoint = None
    latest_numbered_batch = -1

    with open(checkpoints_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                checkpoints.append({
                    "name": data.get("name"),
                    "batch": data.get("batch"),
                    "state_path": data.get("state_path"),
                })
                if data.get("name") == "final":
                    final_checkpoint = data.get("state_path")
                else:
                    # Track the highest numbered checkpoint
                    batch = data.get("batch", 0)
                    if batch > latest_numbered_batch:
                        latest_numbered_batch = batch
                        latest_numbered_checkpoint = data.get("state_path")

    # Use final if available, otherwise use highest numbered
    latest_checkpoint = final_checkpoint or latest_numbered_checkpoint

    return final_checkpoint, latest_checkpoint, checkpoints


def extract_tinker_uuid(checkpoint_path: Optional[str]) -> Optional[str]:
    """Extract the tinker UUID from a checkpoint path."""
    if not checkpoint_path:
        return None
    # Format: tinker://UUID:train:0/weights/final
    match = re.search(r'tinker://([a-f0-9-]+):', checkpoint_path)
    if match:
        return match.group(1)
    return None


def extract_final_reward(metrics_file: Path, task: str) -> Optional[float]:
    """Extract final reward from metrics.jsonl."""
    if not metrics_file.exists():
        return None

    last_line = None
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                last_line = line

    if not last_line:
        return None

    data = json.loads(last_line)

    # Different tasks have different reward keys
    if task == "psychosis":
        # Psychosis uses env/reward/total or similar
        return data.get("env/reward/total") or data.get("env/all/reward/total")
    else:
        # MCQ and wordchain use env/all/reward/total
        return data.get("env/all/reward/total")


def extract_timestamp(log_dir_name: str) -> str:
    """Extract timestamp from log directory name."""
    # Format: 2026-01-12-17-52-18-Qwen-...
    match = re.match(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', log_dir_name)
    if match:
        return match.group(1)
    return log_dir_name[:19]


def parse_log_directory(log_dir: Path, task: str) -> Optional[RunInfo]:
    """Parse a single log directory and extract all relevant information."""
    checkpoints_file = log_dir / "checkpoints.jsonl"
    metrics_file = log_dir / "metrics.jsonl"
    config_file = log_dir / "config.json"
    wandb_dir = log_dir / "wandb"

    # Skip if no checkpoints file
    if not checkpoints_file.exists():
        return None

    # Extract checkpoints
    final_checkpoint, latest_checkpoint, all_checkpoints = extract_checkpoints(checkpoints_file)

    # Extract wandb run ID
    wandb_run_id = extract_wandb_run_id(wandb_dir)
    wandb_url = f"https://wandb.ai/automated-scaffolding/gepa/runs/{wandb_run_id}" if wandb_run_id else None

    # Extract final reward
    final_reward = extract_final_reward(metrics_file, task)

    # Extract config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    # Extract hint type and seed from config or directory name
    hint_type = None
    seed = 0
    base_model = config.get("model_name", "Qwen/Qwen3-30B-A3B")

    if "dataset_builder" in config:
        hint_type = config["dataset_builder"].get("hint_type")
        seed = config["dataset_builder"].get("seed", 0)

    # Try to extract from directory name if not in config
    dir_name = log_dir.name
    if not hint_type:
        # Look for hint type in directory name
        for ht in ["metadata", "sycophancy", "visual-pattern", "grader-hacking", "unethical", "consistency"]:
            if ht in dir_name:
                hint_type = ht
                break

    # Extract seed from directory name
    seed_match = re.search(r'(\d+)seed', dir_name)
    if seed_match:
        seed = int(seed_match.group(1))

    # Extract max_examples from config
    max_examples = None
    if "dataset_builder" in config:
        max_examples = config["dataset_builder"].get("max_examples")

    return RunInfo(
        log_dir=str(log_dir),
        task=task,
        tinker_uuid=extract_tinker_uuid(latest_checkpoint),
        final_checkpoint=final_checkpoint,
        latest_checkpoint=latest_checkpoint,
        all_checkpoints=all_checkpoints,
        wandb_run_id=wandb_run_id,
        wandb_url=wandb_url,
        final_reward=final_reward,
        config=config,
        base_model=base_model,
        hint_type=hint_type,
        seed=seed,
        timestamp=extract_timestamp(log_dir.name),
        max_examples=max_examples,
    )


def get_model_name(run: RunInfo) -> str:
    """Generate the model name for a run.

    Format:
    - MCQ: mcq-{hint_type}-seed{n} (e.g., mcq-visual-pattern-seed0)
    - Psychosis: psychosis-seed{n}
    - Wordchain: wordchain-seed{n}
    """
    if run.task == "mcq":
        hint = run.hint_type or "unknown"
        return f"{run.task}-{hint}-seed{run.seed}"
    else:
        return f"{run.task}-seed{run.seed}"


def format_yaml_entry(run: RunInfo) -> str:
    """Format a run as a YAML entry for tinker_models.yaml.

    Format:
    - Name: mcq-visual-pattern-seed0, psychosis-seed0, wordchain-seed2
    - Description: "MCQ, visual pattern, seed 0, 2026-01-23, 10000 max_examples, reward=0.91: {url}"
    """
    lines = []

    # Parse timestamp to get date (format: 2026-01-12-17-52-18 -> 2026-01-12)
    parts = run.timestamp.split("-")
    if len(parts) >= 3:
        date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
    else:
        date_str = "unknown"

    model_name = get_model_name(run)

    # Format max_examples (null means "all")
    max_examples_str = "all" if run.max_examples is None else str(run.max_examples)

    # Format reward
    reward_str = f"reward={run.final_reward:.2f}" if run.final_reward is not None else "reward=N/A"

    # Generate description based on task type
    if run.task == "mcq":
        hint_display = (run.hint_type or "unknown").replace("-", " ")
        desc = f"MCQ, {hint_display}, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {reward_str}"
    elif run.task == "psychosis":
        desc = f"Psychosis, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {reward_str}"
    elif run.task == "wordchain":
        desc = f"Wordchain, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {reward_str}"
    else:
        desc = f"{run.task}, seed {run.seed}, {date_str}, {max_examples_str} max_examples, {reward_str}"

    if run.wandb_url:
        desc += f": {run.wandb_url}"

    lines.append(f"  {model_name}:")
    lines.append(f"    checkpoint_path: \"{run.latest_checkpoint}\"")
    lines.append(f"    base_model: \"{run.base_model}\"")
    lines.append(f"    description: \"{desc}\"")

    return "\n".join(lines)


def print_summary(all_runs: list[RunInfo]):
    """Print full summary of all runs."""
    print("=" * 120)
    print("TINKER MODEL / WANDB MATCHING SUMMARY")
    print("=" * 120)
    print()

    for task in ["mcq", "psychosis", "wordchain"]:
        task_runs = [r for r in all_runs if r.task == task]
        if not task_runs:
            continue

        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*60}")

        for run in sorted(task_runs, key=lambda r: r.timestamp):
            print()
            print(f"Directory: {os.path.basename(run.log_dir)}")
            print(f"  Timestamp: {run.timestamp}")
            print(f"  Tinker UUID: {run.tinker_uuid}")
            print(f"  Final Checkpoint: {run.final_checkpoint}")
            print(f"  WandB Run ID: {run.wandb_run_id}")
            print(f"  WandB URL: {run.wandb_url}")
            print(f"  Final Reward: {run.final_reward:.4f}" if run.final_reward else "  Final Reward: N/A")
            print(f"  Hint Type: {run.hint_type}")
            print(f"  Seed: {run.seed}")
            print(f"  Base Model: {run.base_model}")

            # Show available checkpoints
            if run.all_checkpoints:
                print(f"  Available Checkpoints: {[c['name'] for c in run.all_checkpoints]}")


def deduplicate_runs(runs: list[RunInfo]) -> list[RunInfo]:
    """Deduplicate runs by model name, keeping the most recent one.

    Special exceptions: certain wandb runs are prioritized regardless of timestamp.
    """
    # Wandb run IDs that should be prioritized over others with the same name
    PRIORITIZED_WANDB_RUNS = {
        "bsp7kgva",  # psychosis-seed2: older but better performing
    }

    # Sort by timestamp descending (most recent first)
    sorted_runs = sorted(runs, key=lambda r: r.timestamp, reverse=True)

    # But put prioritized runs at the front
    prioritized = [r for r in sorted_runs if r.wandb_run_id in PRIORITIZED_WANDB_RUNS]
    non_prioritized = [r for r in sorted_runs if r.wandb_run_id not in PRIORITIZED_WANDB_RUNS]
    sorted_runs = prioritized + non_prioritized

    seen_names = set()
    deduped = []

    for run in sorted_runs:
        name = get_model_name(run)
        if name not in seen_names:
            seen_names.add(name)
            deduped.append(run)

    # Return in alphabetical order by name
    return sorted(deduped, key=lambda r: get_model_name(r))


def print_yaml(all_runs: list[RunInfo]):
    """Print YAML suggestions for tinker_models.yaml."""
    print("# YAML ENTRIES FOR tinker_models.yaml")
    print()

    for task in ["mcq", "psychosis", "wordchain"]:
        # Filter to runs that have at least one checkpoint
        task_runs = [r for r in all_runs if r.task == task and r.latest_checkpoint]
        if not task_runs:
            continue

        # Deduplicate by model name, keeping most recent
        deduped_runs = deduplicate_runs(task_runs)

        # Sort alphabetically by model name
        deduped_runs.sort(key=lambda r: get_model_name(r))

        print(f"\n  # {task.upper()} models")
        for run in deduped_runs:
            print(format_yaml_entry(run))
            print()


def print_json(all_runs: list[RunInfo]):
    """Print JSON output."""
    output = []
    for run in all_runs:
        run_dict = {
            "task": run.task,
            "timestamp": run.timestamp,
            "tinker_uuid": run.tinker_uuid,
            "final_checkpoint": run.final_checkpoint,
            "wandb_run_id": run.wandb_run_id,
            "wandb_url": run.wandb_url,
            "final_reward": run.final_reward,
            "hint_type": run.hint_type,
            "seed": run.seed,
            "base_model": run.base_model,
            "checkpoints": [c["name"] for c in run.all_checkpoints],
            "log_dir": run.log_dir,
        }
        output.append(run_dict)
    print(json.dumps(output, indent=2))


def print_csv(all_runs: list[RunInfo]):
    """Print CSV output."""
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "task", "timestamp", "tinker_uuid", "final_checkpoint",
        "wandb_run_id", "wandb_url", "final_reward", "hint_type",
        "seed", "base_model", "checkpoints", "log_dir"
    ])
    for run in all_runs:
        writer.writerow([
            run.task,
            run.timestamp,
            run.tinker_uuid or "",
            run.final_checkpoint or "",
            run.wandb_run_id or "",
            run.wandb_url or "",
            f"{run.final_reward:.4f}" if run.final_reward else "",
            run.hint_type or "",
            run.seed,
            run.base_model,
            ";".join(c["name"] for c in run.all_checkpoints),
            run.log_dir,
        ])


def main():
    parser = argparse.ArgumentParser(description="Match tinker models with wandb runs")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--csv", action="store_true",
                        help="Output as CSV")
    parser.add_argument("--yaml", action="store_true",
                        help="Output YAML entries for tinker_models.yaml")
    parser.add_argument("--task", choices=["mcq", "psychosis", "wordchain"],
                        help="Filter by task type")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-30B-A3B",
                        help="Filter by base model (default: Qwen/Qwen3-30B-A3B, use 'all' for no filter)")
    parser.add_argument("--logs-dir", type=Path,
                        default=Path("/nas/ucb/biddulph/gepa-legibility/logs-rl"),
                        help="Path to logs-rl directory")
    args = parser.parse_args()

    logs_rl_dir = args.logs_dir
    all_runs = []

    tasks = [args.task] if args.task else ["mcq", "psychosis", "wordchain"]

    for task in tasks:
        task_dir = logs_rl_dir / task
        if not task_dir.exists():
            continue

        for log_dir in sorted(task_dir.iterdir()):
            if log_dir.is_dir():
                run_info = parse_log_directory(log_dir, task)
                if run_info:
                    all_runs.append(run_info)

    # Filter by base model (unless 'all' specified)
    if args.base_model != "all":
        all_runs = [r for r in all_runs if r.base_model == args.base_model]

    # Sort by timestamp
    all_runs.sort(key=lambda r: (r.task, r.timestamp))

    # Output in requested format
    if args.json:
        print_json(all_runs)
    elif args.csv:
        print_csv(all_runs)
    elif args.yaml:
        print_yaml(all_runs)
    else:
        print_summary(all_runs)
        print()
        print_yaml(all_runs)


if __name__ == "__main__":
    main()
