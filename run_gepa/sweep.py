#!/usr/bin/env python3
"""Sweep runner for GEPA experiments with multiple configurations.

Usage:
    # Fresh run
    python -m run_gepa.sweep sweeps/my_experiment.yaml

    # Resume an existing run
    python -m run_gepa.sweep --resume logs/mcq/my_experiment/2024-12-04-10-30-00/
"""

import argparse
import itertools
import traceback
from datetime import datetime
from pathlib import Path

import yaml


# Fields that are never swept over (always scalar)
NON_SWEEPABLE_FIELDS = {"name", "task", "num_trials", "cache", "num_threads", "date_str", "zip_fields"}

# Required fields for all experiments (including baseline-only)
BASE_REQUIRED_FIELDS = [
    "prompter_name",  # Can be null for baseline-only
    "executor_name",
    "incompetent",
    "num_threads",
]

# Additional fields required only for GEPA optimization (when prompter_name is not null)
GEPA_REQUIRED_FIELDS = [
    "suggest_hack",
    "max_metric_calls",
    "validation_set_size",
    "length_penalty",
    "verbalization_penalty",
    "baseline_instructions_path",
    "executor_reasoning_effort",
    "reflection_minibatch_size",
]

# Additional required fields specific to each task (for GEPA only)
TASK_GEPA_REQUIRED_FIELDS = {
    "mcq": [],  # hint_type/datamix required for both baseline and GEPA
    "psychosis": ["train_teacher_file"],
    "wordchain": ["train_teacher_file"],
}

# Task-specific fields required for all experiments (including baseline-only)
TASK_BASE_REQUIRED_FIELDS = {
    "mcq": ["hint_type", "datamix"],
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
    if task not in TASK_BASE_REQUIRED_FIELDS:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(TASK_BASE_REQUIRED_FIELDS.keys())}")

    # Determine if this is baseline-only (prompter_name is null/None)
    is_baseline_only = config.get("prompter_name") is None

    # Build required fields based on mode
    required_fields = BASE_REQUIRED_FIELDS + TASK_BASE_REQUIRED_FIELDS[task]
    if not is_baseline_only:
        required_fields = required_fields + GEPA_REQUIRED_FIELDS + TASK_GEPA_REQUIRED_FIELDS[task]

    # Validate required fields
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Missing required fields for task '{task}': {missing}")

    # Set defaults for optional fields
    config.setdefault("num_trials", 1)
    config.setdefault("cache", False)
    config.setdefault("teacher_tool_model", None)
    config.setdefault("show_expert_reasoning", False)

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
    # Path structure: logs/<task>/<name>/<datestr>/
    date_str = resume_path.name
    experiment_name = resume_path.parent.name

    return config, experiment_name, date_str


def _parse_zip_fields(config: dict) -> tuple[list[list[str]], set[str]]:
    """Parse and validate zip_fields from config.

    Returns:
        tuple of (zip_groups, set of all zipped field names)

    Raises:
        ValueError: If zip_fields references non-existent fields, non-list fields,
            or fields with mismatched lengths within a group.
    """
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
            raise ValueError(f"zip_fields group {group} has mismatched lengths: {lengths}")

    return zip_groups, zipped_fields


def get_sweep_combinations(config: dict) -> tuple[list[str], list[tuple]]:
    """Extract sweep dimensions and generate all combinations.

    Supports zip_fields for paired sweeps. When zip_fields is specified, the listed
    fields are zipped together (swept in lockstep) rather than cross-producted.

    Example config with zip_fields:
        zip_fields:
          - [hint_type, executor_name]  # These sweep together
        hint_type: [a, b, c]
        executor_name: [1, 2, 3]
        other_field: [x, y]

    This produces 6 combinations (3 paired × 2 other), not 18 (3 × 3 × 2):
        (a, 1, x), (a, 1, y), (b, 2, x), (b, 2, y), (c, 3, x), (c, 3, y)

    Returns:
        tuple of (field_names, list of value tuples)
    """
    zip_groups, zipped_fields = _parse_zip_fields(config)

    # Build sweep dimensions:
    # - Zipped groups become a single dimension with tuples of values
    # - Non-zipped list fields become individual dimensions
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
        tuple(v for item in combo for v in (item if isinstance(item, tuple) else (item,)))
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
    exp_dir = Path(f"logs/{task}/{experiment_name}/{date_str}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_sweep_config(exp_dir: Path, config: dict) -> None:
    """Save the sweep config to the experiment directory."""
    config_path = exp_dir / "sweep_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved sweep config to {config_path}")


def run_sweep(config: dict, experiment_name: str, date_str: str) -> None:
    """Run all combinations in the sweep."""
    task = config["task"]
    num_trials = config["num_trials"]  # Default set in load_config

    # Get sweep dimensions and fixed values
    sweep_fields, combinations = get_sweep_combinations(config)
    fixed_values = get_fixed_values(config)

    print(f"Task: {task}")
    print(f"Experiment: {experiment_name}")
    print(f"Date: {date_str}")
    print(f"Sweep dimensions: {sweep_fields}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Trials per combination: {num_trials}")
    print(f"Total runs: {len(combinations) * num_trials}")
    print()

    # Import the appropriate task runner
    if task == "mcq":
        from run_gepa.mcq import run_single_mcq_trial
        task_runner = run_single_mcq_trial
    elif task == "psychosis":
        from run_gepa.psychosis import run_single_psychosis_trial
        task_runner = run_single_psychosis_trial
    elif task == "wordchain":
        from run_gepa.wordchain import run_single_wordchain_trial
        task_runner = run_single_wordchain_trial
    else:
        raise ValueError(f"Unknown task: {task}")

    # Run all combinations
    run_count = 0
    for trial_idx in range(num_trials):
        for combo in combinations:
            # Build kwargs for this combination
            combo_kwargs = dict(zip(sweep_fields, combo))
            all_kwargs = {**fixed_values, **combo_kwargs}

            run_count += 1
            print(f"\n{'='*60}")
            print(f"Run {run_count}/{len(combinations) * num_trials}")
            print(f"Config: {combo_kwargs}")
            print(f"Trial: {trial_idx}")
            print(f"{'='*60}")

            try:
                task_runner(
                    **all_kwargs,
                    experiment_name=experiment_name,
                    date_str=date_str,
                    trial_idx=trial_idx,
                    cache=config.get("cache", False),
                    sweep_fields=sweep_fields,
                )
            except Exception:
                traceback.print_exc()
                # Continue with other runs


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA sweeps over multiple configurations"
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to sweep config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from an existing experiment directory (e.g., logs/mcq/my_exp/2024-12-04/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually running",
    )

    args = parser.parse_args()

    if args.resume:
        # Resume mode: load config from existing experiment
        config, experiment_name, date_str = load_config_from_resume(args.resume)
        print(f"Resuming experiment from {args.resume}")
    elif args.config_path:
        # Fresh run: load config from YAML
        config = load_config(args.config_path)
        experiment_name = config["name"]
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        parser.error("Either config_path or --resume must be provided")

    if args.dry_run:
        sweep_fields, combinations = get_sweep_combinations(config)
        fixed_values = get_fixed_values(config)
        num_trials = config.get("num_trials", 1)

        print("DRY RUN - Would execute:")
        print(f"  Task: {config['task']}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Sweep fields: {sweep_fields}")
        print(f"  Fixed values: {fixed_values}")
        print(f"  Combinations: {len(combinations)}")
        print(f"  Trials per combo: {num_trials}")
        print(f"  Total runs: {len(combinations) * num_trials}")
        print("\nCombinations:")
        for i, combo in enumerate(combinations):
            combo_dict = dict(zip(sweep_fields, combo))
            print(f"  {i+1}. {combo_dict}")
        return

    # Create experiment directory and save config (only for fresh runs, not resume)
    if not args.resume:
        exp_dir = make_experiment_dir(config["task"], experiment_name, date_str)
        save_sweep_config(exp_dir, config)

    run_sweep(config, experiment_name, date_str)


if __name__ == "__main__":
    main()
