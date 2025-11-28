"""
DataFrame loader for GEPA experiment data.

This module provides utilities to load experiment progression data into a pandas DataFrame
for flexible analysis and plotting.
"""

import json
from pathlib import Path
import re

import pandas as pd

from core.progression.loader import get_progression_data


def load_experiment_df(
    paths: list[str],
    quick_mode: bool = False,
) -> pd.DataFrame:
    """Load experiment data from multiple paths into a single DataFrame.

    Args:
        paths: List of paths to experiment directories. Can be:
            - Timestamp level: "logs/mcq/2025-11-19-11-41-43/" (loads all experiments under it)
            - Specific: "logs/mcq/2025-11-19-11-41-43/p=claude-..." (loads just that one)
        quick_mode: If True, only evaluate first+last candidates (for bar plots).
                   If False, evaluate all improving candidates (for progression plots).

    Returns:
        DataFrame with one row per (experiment, run, candidate, is_sanitized, subset, metric_name).
    """
    # Expand paths to specific experiment directories
    expanded_paths = _expand_paths(paths)

    all_rows = []
    for exp_path in expanded_paths:
        rows = _load_single_experiment(exp_path, quick_mode)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No data found in the specified paths")

    return pd.DataFrame(all_rows)


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
            raise ValueError(f"Invalid path (expected timestamp directory or its child): {path}")

    return expanded


def _is_timestamp_path(path: Path) -> bool:
    """Check if a path is a timestamp directory."""
    if not path.is_dir():
        return False

    return re.match(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$', path.name) is not None


def _load_single_experiment(exp_path: Path, quick_mode: bool) -> list[dict]:
    """Load data for a single experiment path into rows.

    Args:
        exp_path: Path to specific experiment directory
        quick_mode: Whether to use quick mode for loading

    Returns:
        List of row dicts
    """
    # Load progression data (this handles caching and evaluation)
    progression_data = get_progression_data(exp_path, quick_mode=quick_mode)

    # Determine if this is MCQ (has hint_type structure)
    if isinstance(progression_data, dict) and "runs" not in progression_data:
        # MCQ: progression_data is {hint_type: {"runs": [...]}}
        rows = []
        for hint_type, hint_data in progression_data.items():
            hint_path = exp_path / hint_type
            rows.extend(_flatten_runs(hint_data, hint_path))
        return rows
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
        print(f"  Warning: env_name not in config.json, inferred '{environment}' from path")

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
    raise ValueError(f"Cannot infer environment from path '{path_str}', expected one of {KNOWN_ENVIRONMENTS} in path")


def _extract_short_model_name(full_name: str) -> str:
    """Extract short model name from full provider/model string."""
    if not full_name:
        return ""
    # Handle formats like "anthropic/claude-sonnet-4-5-20250929" or "deepinfra/Qwen/Qwen3-14B"
    parts = full_name.split("/")
    return parts[-1] if parts else full_name


def _flatten_runs(progression_data: dict, path: Path) -> list[dict]:
    """Flatten progression data to rows.

    Args:
        progression_data: Dict with "runs" key containing list of run data
        path: Path to load config from (experiment path for non-MCQ, hint_type path for MCQ)

    Returns:
        List of row dicts
    """
    config = _load_config(path)
    metadata = _extract_metadata(path, config)
    rows = []

    for run_data in progression_data["runs"]:
        run_index = run_data["run_index"]
        progression = run_data["progression"]

        # Find the best validation score to mark is_final
        best_val_score = max(p["validation_score"] for p in progression) if progression else 0

        for prog_point in progression:
            candidate_index = prog_point["candidate_index"]
            is_baseline = candidate_index == 0
            is_final = prog_point["validation_score"] == best_val_score

            # Create base row data
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
                metric_type, is_sanitized = _parse_score_key(score_key)
                rows.append({
                    **base_row,
                    "is_sanitized": is_sanitized,
                    "subset": None,
                    "metric_name": f"{metric_type}_reward",
                    "metric_value": score_value,
                })

            # Expand subset_test_scores into rows
            for subset_name, subset_scores in prog_point.get("subset_test_scores", {}).items():
                for score_key, score_value in subset_scores.items():
                    metric_type, is_sanitized = _parse_score_key(score_key)
                    rows.append({
                        **base_row,
                        "is_sanitized": is_sanitized,
                        "subset": subset_name,
                        "metric_name": f"{metric_type}_reward",
                        "metric_value": score_value,
                    })

    return rows


VALID_METRIC_TYPES = {"proxy", "true"}
VALID_SANITIZATION_TYPES = {"original", "sanitized"}


def _parse_score_key(score_key: str) -> tuple[str, bool]:
    """Parse score key like 'proxy_original' into (metric_type, is_sanitized)."""
    parts = score_key.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid score key format: '{score_key}', expected 'metric_sanitization' (e.g., 'proxy_original')")

    metric_type, sanitization_type = parts
    if metric_type not in VALID_METRIC_TYPES:
        raise ValueError(f"Invalid metric type '{metric_type}' in score key '{score_key}', expected one of {VALID_METRIC_TYPES}")
    if sanitization_type not in VALID_SANITIZATION_TYPES:
        raise ValueError(f"Invalid sanitization type '{sanitization_type}' in score key '{score_key}', expected one of {VALID_SANITIZATION_TYPES}")

    return metric_type, sanitization_type == "sanitized"
