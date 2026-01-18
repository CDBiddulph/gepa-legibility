#!/usr/bin/env python3
"""Sweep runner for prompt shortening experiments.

Usage:
    # Fresh run
    python -m run_shortening.sweep sweeps/shorten-example.yaml

    # Resume an existing run
    python -m run_shortening.sweep --resume logs/shortening/wordchain/my_exp/2024-12-04/
"""

import argparse
import hashlib
import itertools
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from run_shortening.shorten import ShorteningConfig, run_shortening


# Fields that are never swept over (always scalar)
NON_SWEEPABLE_FIELDS = {"name", "task", "cache", "num_threads", "date_str", "zip_fields"}

# Required fields for shortening experiments
REQUIRED_FIELDS = [
    "name",
    "task",
    "prompter_name",
    "executor_name",
    "baseline_instructions_path",
    "max_iterations",
    "num_threads",
]

# Task-specific required fields
TASK_REQUIRED_FIELDS = {
    "mcq": ["hint_type"],
    "psychosis": [],
    "wordchain": [],
}


def load_config(config_path: str) -> dict:
    """Load and validate a sweep configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required top-level fields
    if "name" not in config:
        raise ValueError("Config must have a 'name' field")
    if "task" not in config:
        raise ValueError("Config must have a 'task' field")

    task = config["task"]
    if task not in TASK_REQUIRED_FIELDS:
        raise ValueError(
            f"Unknown task: {task}. Must be one of {list(TASK_REQUIRED_FIELDS.keys())}"
        )

    # Build required fields
    required_fields = REQUIRED_FIELDS + TASK_REQUIRED_FIELDS[task]

    # Validate required fields
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Missing required fields for task '{task}': {missing}")

    # Set defaults
    config.setdefault("cache", False)

    return config


def load_config_from_resume(resume_path: str) -> tuple[dict, str, str]:
    """Load config from a resumed experiment directory.

    Returns:
        tuple of (config, experiment_name, date_str)
    """
    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise ValueError(f"Resume path does not exist: {resume_path}")

    config_path = resume_path / "sweep_config.yaml"
    if not config_path.exists():
        raise ValueError(f"No sweep_config.yaml found in {resume_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract experiment_name and date_str from path
    # Path structure: logs/shortening/<task>/<name>/<datestr>/
    date_str = resume_path.name
    experiment_name = resume_path.parent.name

    return config, experiment_name, date_str


def _parse_zip_fields(config: dict) -> tuple[list[list[str]], set[str]]:
    """Parse and validate zip_fields from config."""
    zip_groups = config.get("zip_fields", [])
    zipped_fields = set()
    for group in zip_groups:
        zipped_fields.update(group)

    for group in zip_groups:
        lengths = {}
        for field in group:
            if field not in config:
                raise ValueError(f"zip_fields references non-existent field: {field}")
            if not isinstance(config[field], list):
                raise ValueError(f"zip_fields field must be a list: {field}")
            lengths[field] = len(config[field])
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"zip_fields group {group} has mismatched lengths: {lengths}"
            )

    return zip_groups, zipped_fields


def get_sweep_combinations(config: dict) -> tuple[list[str], list[tuple]]:
    """Extract sweep dimensions and generate all combinations."""
    zip_groups, zipped_fields = _parse_zip_fields(config)

    sweep_fields = []
    sweep_values = []

    # Add zipped groups as combined dimensions
    for group in zip_groups:
        sweep_fields.extend(group)
        group_values = [config[field] for field in group]
        sweep_values.append(list(zip(*group_values)))

    # Add non-zipped list fields
    for key, value in config.items():
        if key in NON_SWEEPABLE_FIELDS or key in zipped_fields:
            continue
        if isinstance(value, list):
            sweep_fields.append(key)
            sweep_values.append([(v,) for v in value])

    if not sweep_values:
        return [], [()]

    # Cross-product all dimensions and flatten nested tuples
    combinations = [
        tuple(
            v for item in combo for v in (item if isinstance(item, tuple) else (item,))
        )
        for combo in itertools.product(*sweep_values)
    ]

    return sweep_fields, combinations


def get_fixed_values(config: dict) -> dict:
    """Extract non-sweep (scalar) values from config."""
    fixed = {}
    for key, value in config.items():
        if key in NON_SWEEPABLE_FIELDS:
            continue
        if not isinstance(value, list):
            fixed[key] = value
    return fixed


def make_experiment_dir(task: str, experiment_name: str, date_str: str) -> Path:
    """Create the experiment directory structure."""
    exp_dir = Path(f"logs/shortening/{task}/{experiment_name}/{date_str}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_sweep_config(exp_dir: Path, config: dict) -> None:
    """Save the sweep config to the experiment directory."""
    config_path = exp_dir / "sweep_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved sweep config to {config_path}")


def _sanitize_value(value) -> str:
    """Sanitize a value for use in a directory name."""
    s = str(value)
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    return s


# Fields that should use a hash instead of the full value (to avoid long paths)
HASH_FIELDS = {"baseline_instructions_path"}


def _format_swept_path(sweep_fields: list[str], combo_kwargs: dict) -> str:
    """Generate path component based on swept parameters."""
    if not sweep_fields:
        return "default"

    parts = []
    for field in sweep_fields:
        value = combo_kwargs[field]
        if field in HASH_FIELDS:
            # Use a short hash for fields with long values
            hash_val = hashlib.sha256(str(value).encode()).hexdigest()[:12]
            parts.append(f"{field}={hash_val}")
        else:
            sanitized = _sanitize_value(value)
            parts.append(f"{field}={sanitized}")

    return "-".join(parts)


def run_sweep(config: dict, experiment_name: str, date_str: str) -> None:
    """Run all combinations in the sweep."""
    task = config["task"]

    # Get sweep dimensions and fixed values
    sweep_fields, combinations = get_sweep_combinations(config)
    fixed_values = get_fixed_values(config)

    print(f"Task: {task}")
    print(f"Experiment: {experiment_name}")
    print(f"Date: {date_str}")
    print(f"Sweep dimensions: {sweep_fields}")
    print(f"Total combinations: {len(combinations)}")
    print()

    # Run all combinations
    run_count = 0
    for combo in combinations:
        # Build kwargs for this combination
        combo_kwargs = dict(zip(sweep_fields, combo))
        all_kwargs = {**fixed_values, **combo_kwargs}

        run_count += 1
        print(f"\n{'='*60}")
        print(f"Run {run_count}/{len(combinations)}")
        print(f"Config: {combo_kwargs}")
        print(f"{'='*60}")

        # Build log directory
        exp_dir = Path(f"logs/shortening/{task}/{experiment_name}/{date_str}")
        varied_path = _format_swept_path(sweep_fields, combo_kwargs)
        log_dir = str(exp_dir / varied_path / "0")

        # Create ShorteningConfig
        shortening_config = ShorteningConfig(
            prompter_name=all_kwargs["prompter_name"],
            executor_name=all_kwargs["executor_name"],
            baseline_instructions_path=all_kwargs["baseline_instructions_path"],
            max_iterations=all_kwargs["max_iterations"],
            env_name=task,
            num_threads=config["num_threads"],
            validation_set_size=config.get("validation_set_size"),
            date_str=date_str,
            experiment_name=experiment_name,
            sweep_fields=sweep_fields,
            log_dir_index=0,
            cache=config.get("cache", False),
            hint_type=all_kwargs.get("hint_type"),
            incompetent=config.get("incompetent", False),
            evaluate_single_piece_removals=config.get("evaluate_single_piece_removals", False),
            num_proposals_per_iteration=config.get("num_proposals_per_iteration", 1),
        )

        try:
            run_shortening(shortening_config, log_dir)
        except Exception:
            traceback.print_exc()
            # Continue with other runs


def main():
    parser = argparse.ArgumentParser(
        description="Run shortening sweeps over multiple configurations"
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to sweep config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from an existing experiment directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually running",
    )

    args = parser.parse_args()

    if args.resume:
        config, experiment_name, date_str = load_config_from_resume(args.resume)
        print(f"Resuming experiment from {args.resume}")
    elif args.config_path:
        config = load_config(args.config_path)
        experiment_name = config["name"]
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        parser.error("Either config_path or --resume must be provided")

    if args.dry_run:
        sweep_fields, combinations = get_sweep_combinations(config)
        fixed_values = get_fixed_values(config)

        print("DRY RUN - Would execute:")
        print(f"  Task: {config['task']}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Sweep fields: {sweep_fields}")
        print(f"  Fixed values: {fixed_values}")
        print(f"  Combinations: {len(combinations)}")
        print("\nCombinations:")
        for i, combo in enumerate(combinations):
            combo_dict = dict(zip(sweep_fields, combo))
            print(f"  {i+1}. {combo_dict}")
        return

    # Create experiment directory and save config (only for fresh runs)
    if not args.resume:
        exp_dir = make_experiment_dir(config["task"], experiment_name, date_str)
        save_sweep_config(exp_dir, config)

    run_sweep(config, experiment_name, date_str)


if __name__ == "__main__":
    main()
