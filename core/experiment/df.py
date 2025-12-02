"""
DataFrame loader for GEPA experiment data.

This module provides utilities to load experiment progression data into DataFrames
for flexible analysis and plotting.
"""

import json
from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

from core.evaluation import get_progression_data
from core.metrics import METRICS_BY_PROMPT_VERSION, METRICS


# Type alias for the dictionary of DataFrames
DFs = dict[str, pd.DataFrame]


def load_experiment_dfs(
    paths: list[str],
    quick_mode: bool = False,
) -> DFs:
    """Load experiment data from multiple paths into DataFrames.

    Args:
        paths: List of paths to experiment directories. Can be:
            - Timestamp level: "logs/mcq/2025-11-19-11-41-43/" (loads all experiments under it)
            - Specific: "logs/mcq/2025-11-19-11-41-43/p=claude-..." (loads just that one)
        quick_mode: If True, only evaluate first+last candidates (for bar plots).
                   If False, evaluate all improving candidates (for progression plots).

    Returns:
        Dict with two DataFrames:
        - "main": One row per (experiment, run, candidate, is_sanitized, subset, metric_name)
        - "gepa_runs": One row per (experiment, run) with run-level info like end state
    """
    # Expand paths to specific experiment directories
    expanded_paths = _expand_paths(paths)

    all_rows = defaultdict(list)
    for exp_path in expanded_paths:
        rows = _load_single_experiment(exp_path, quick_mode)
        for key in rows:
            all_rows[key].extend(rows[key])

    if not all_rows.get("main"):
        raise ValueError("No data found in the specified paths")

    return {key: pd.DataFrame(rows) for key, rows in all_rows.items()}


def _expand_paths(paths: list[str]) -> list[Path]:
    """Expand timestamp-level paths to specific experiment paths.

    Args:
        paths: List of paths (can be timestamp-level or specific)

    Returns:
        List of specific experiment directory paths
    """
    expanded = []
    for path_str in paths:
        path = Path(path_str)

        # Check if this is a timestamp-level path or a specific experiment path
        # by looking at whether the parent looks like a timestamp
        if _is_timestamp_path(path):
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir():
                    expanded.append(subdir)
        elif _is_timestamp_path(path.parent):
            expanded.append(path)
        else:
            raise ValueError(
                f"Invalid path (expected timestamp directory or its child): {path}"
            )

    return expanded


def _is_timestamp_path(path: Path) -> bool:
    """Check if a path is a timestamp directory."""
    if not path.is_dir():
        return False

    return re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", path.name) is not None


def _load_single_experiment(
    exp_path: Path, quick_mode: bool
) -> dict[str, list[dict]]:
    """Load data for a single experiment path into rows.

    Args:
        exp_path: Path to specific experiment directory
        quick_mode: Whether to use quick mode for loading

    Returns:
        Dict with "main" and "gepa_runs" keys containing row lists
    """
    # Load progression data (this handles caching and evaluation)
    progression_data = get_progression_data(exp_path, quick_mode=quick_mode)

    # Determine if this is MCQ (has hint_type structure)
    if isinstance(progression_data, dict) and "runs" not in progression_data:
        # MCQ: progression_data is {hint_type: {"runs": [...]}}
        all_rows = defaultdict(list)
        for hint_type, hint_data in progression_data.items():
            hint_path = exp_path / hint_type
            rows = _flatten_runs(hint_data, hint_path)
            for key in rows:
                all_rows[key].extend(rows[key])
        return all_rows
    else:
        # Non-MCQ: progression_data is {"runs": [...]}
        return _flatten_runs(progression_data, exp_path)


def _load_config(path: Path) -> dict:
    """Load config.json from the first run directory under path."""
    config_files = sorted(path.glob("*/config.json"))
    if not config_files:
        raise FileNotFoundError(f"No config.json found in {path}")
    with open(config_files[0]) as f:
        return json.load(f)


def _extract_metadata(exp_path: Path, config: dict) -> dict:
    """Extract metadata from path and config."""
    path_str = str(exp_path)

    # Extract timestamp from path (format: logs/{env}/{timestamp}/{params})
    parts = exp_path.parts
    # Find the timestamp part (looks like 2025-11-19-11-41-43)
    timestamp = None
    for part in parts:
        if len(part) == 19 and part.count("-") == 5:  # YYYY-MM-DD-HH-MM-SS
            timestamp = part
            break

    # Get environment from config, falling back to path inference
    if "env_name" in config:
        environment = config["env_name"]
    else:
        environment = _infer_environment(path_str)
        print(
            f"  Warning: env_name not in config.json, inferred '{environment}' from path"
        )

    return {
        "experiment_path": path_str,
        "environment": environment,
        "prompter_model": _extract_short_model_name(config["prompter_name"]),
        "executor_model": _extract_short_model_name(config["executor_name"]),
        "hack_setting": config["suggest_hack"],
        "hint_type": config.get("hint_type"),  # None for non-MCQ
        "timestamp": timestamp,
    }


KNOWN_ENVIRONMENTS = {"psychosis", "mcq", "wordchain", "alliterate"}


def _infer_environment(path_str: str) -> str:
    """Infer environment from path if not in config."""
    for env in KNOWN_ENVIRONMENTS:
        if env in path_str:
            return env
    raise ValueError(
        f"Cannot infer environment from path '{path_str}', expected one of {KNOWN_ENVIRONMENTS} in path"
    )


def _extract_short_model_name(full_name: str) -> str:
    """Extract short model name from full provider/model string."""
    if not full_name:
        return ""
    # Handle formats like "anthropic/claude-sonnet-4-5-20250929" or "deepinfra/Qwen/Qwen3-14B"
    parts = full_name.split("/")
    return parts[-1] if parts else full_name


def _flatten_runs(progression_data: dict, path: Path) -> dict[str, list[dict]]:
    """Flatten progression data to rows.

    Args:
        progression_data: Dict with "runs" key containing list of run data
        path: Path to load config from (experiment path for non-MCQ, hint_type path for MCQ)

    Returns:
        Dict with "main" and "gepa_runs" keys where:
        - main: Candidate-level data (one row per candidate/metric/subset)
        - gepa_runs: Run-level data (one row per run with end state info)
    """
    config = _load_config(path)
    metadata = _extract_metadata(path, config)
    rows = {"main": [], "gepa_runs": []}

    for run_data in progression_data["runs"]:
        run_index = run_data["run_index"]
        progression = run_data["progression"]

        # Find the best validation score to mark is_final
        best_val_score = (
            max(p["validation_score"] for p in progression) if progression else 0
        )

        for prog_point in progression:
            candidate_index = prog_point["candidate_index"]
            is_baseline = candidate_index == 0
            is_final = prog_point["validation_score"] == best_val_score

            # Create base row data for main DataFrame
            base_row = {
                **metadata,
                "run_index": run_index,
                "candidate_index": candidate_index,
                "is_baseline": is_baseline,
                "is_final": is_final,
                "validation_score": prog_point["validation_score"],
                "discovery_eval_counts": prog_point["discovery_eval_counts"],
                "reflection_call_count": prog_point["reflection_call_count"],
            }

            # Expand test_scores into rows (overall scores, subset=None)
            for score_key, score_value in prog_point["test_scores"].items():
                metric_name, is_sanitized = _parse_score_key(score_key)
                rows["main"].append(
                    {
                        **base_row,
                        "is_sanitized": is_sanitized,
                        "subset": None,
                        "metric_name": metric_name,
                        "metric_value": score_value,
                    }
                )

            # Expand subset_test_scores into rows
            for subset_name, subset_scores in prog_point.get(
                "subset_test_scores", {}
            ).items():
                for score_key, score_value in subset_scores.items():
                    metric_name, is_sanitized = _parse_score_key(score_key)
                    rows["main"].append(
                        {
                            **base_row,
                            "is_sanitized": is_sanitized,
                            "subset": subset_name,
                            "metric_name": metric_name,
                            "metric_value": score_value,
                        }
                    )

        # Add run-level row to gepa_runs DataFrame
        end_state = run_data["end_state"]
        rows["gepa_runs"].append(
            {
                **metadata,
                "run_index": run_index,
                "end_discovery_eval_counts": end_state["discovery_eval_counts"],
                "end_reflection_call_count": end_state["reflection_call_count"],
            }
        )

    return rows


def _parse_score_key(score_key: str) -> tuple[str, bool]:
    """Parse score key like 'proxy_reward_original' into (metric_name, is_sanitized)."""
    # Split from the right to handle metric names with underscores (e.g., "proxy_reward")
    # We assume that the prompt version has no underscores in it.
    parts = score_key.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid score key format: '{score_key}', expected 'metric_version' (e.g., 'proxy_reward_original')"
        )

    valid_prompt_versions = set(METRICS_BY_PROMPT_VERSION.keys())
    valid_metric_names = set(METRICS.keys())

    metric_name, prompt_version = parts
    if metric_name not in valid_metric_names:
        raise ValueError(
            f"Invalid metric name '{metric_name}' in score key '{score_key}', expected one of {valid_metric_names}"
        )
    if prompt_version not in valid_prompt_versions:
        raise ValueError(
            f"Invalid prompt version '{prompt_version}' in score key '{score_key}', expected one of {valid_prompt_versions}"
        )
    if len(valid_prompt_versions) != 2:
        raise ValueError(
            f"Expected exactly two prompt versions to return boolean is_sanitized, got {valid_prompt_versions}"
        )

    return metric_name, prompt_version == "sanitized"
