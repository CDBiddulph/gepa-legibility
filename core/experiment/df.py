"""
DataFrame loader for GEPA experiment data.

This module provides utilities to load experiment progression data into DataFrames
for flexible analysis and plotting.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.evaluation import get_progression_data
from core.experiment_utils import expand_experiment_paths
from core.metrics import METRICS_BY_PROMPT_VERSION, METRICS
from core.shortening.results import load_shortening_results


# Type alias for the dictionary of DataFrames
DFs = dict[str, pd.DataFrame]


def load_experiment_dfs(
    paths: list[str],
    quick_mode: bool = False,
    metrics: list[str] | None = None,
) -> DFs:
    """Load experiment data from multiple paths into DataFrames.

    Args:
        paths: List of paths to experiment directories. Can be:
            - Timestamp level: "logs/mcq/2025-11-19-11-41-43/" (loads all experiments under it)
            - Specific: "logs/mcq/2025-11-19-11-41-43/p=claude-..." (loads just that one)
        quick_mode: If True, only evaluate first+last candidates (for bar plots).
                   If False, evaluate all improving candidates (for progression plots).
        metrics: List of metric names to compute. If None, computes all metrics.

    Returns:
        Dict with two DataFrames:
        - "main": One row per (experiment, run, candidate, is_sanitized) with metric columns.
            Overall metrics use column names like "proxy_reward".
            Subset-specific metrics use column names like "proxy_reward_hinted", "hacking_rate_gameable".
        - "gepa_runs": One row per (experiment, run) with run-level info like end state
    """
    # Expand paths to specific experiment directories
    expanded_paths = _expand_paths(paths)

    all_rows = defaultdict(list)
    for exp_path in expanded_paths:
        rows = _load_single_experiment(exp_path, quick_mode, metrics)
        for key in rows:
            all_rows[key].extend(rows[key])

    if not all_rows.get("main"):
        raise ValueError("No data found in the specified paths")

    return {key: pd.DataFrame(rows) for key, rows in all_rows.items()}


def _expand_paths(paths: list[str]) -> list[Path]:
    """Expand paths to specific experiment directories.

    Args:
        paths: List of paths to experiment directories or parent directories

    Returns:
        List of specific experiment directory paths
    """
    expanded = []
    # Filter out None paths, so we can use .get() in .ipynb and ignore absent paths
    for path_str in [p for p in paths if p is not None]:
        expanded.extend(expand_experiment_paths(path_str))
    return expanded


def _load_single_experiment(
    exp_path: Path, quick_mode: bool, metrics: list[str] | None = None
) -> dict[str, list[dict]]:
    """Load data for a single experiment path into rows.

    Args:
        exp_path: Path to specific experiment directory
        quick_mode: Whether to use quick mode for loading
        metrics: List of metric names to compute. If None, computes all metrics.

    Returns:
        Dict with "main" and "gepa_runs" keys containing row lists
    """
    # Load progression data (this handles caching and evaluation)
    progression_data = get_progression_data(exp_path, quick_mode=quick_mode, metrics=metrics)
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
    # Start with path-derived field
    metadata = {"experiment_path": str(exp_path)}

    # Add all config fields (convert lists to tuples for DataFrame hashability)
    for key, value in config.items():
        metadata[key] = tuple(value) if isinstance(value, list) else value

    # Infer env_name from path if not in config
    if "env_name" not in metadata:
        metadata["env_name"] = _infer_environment(str(exp_path))
        print(
            f"  Warning: env_name not in config.json, inferred '{metadata['env_name']}' from path"
        )

    return metadata


KNOWN_ENVIRONMENTS = {"psychosis", "mcq", "wordchain"}


def _infer_environment(path_str: str) -> str:
    """Infer environment from path if not in config."""
    for env in KNOWN_ENVIRONMENTS:
        if env in path_str:
            return env
    raise ValueError(
        f"Cannot infer environment from path '{path_str}', expected one of {KNOWN_ENVIRONMENTS} in path"
    )


def _flatten_runs(progression_data: dict, path: Path) -> dict[str, list[dict]]:
    """Flatten progression data to rows.

    Args:
        progression_data: Dict with "runs" key containing list of run data
        path: Path to load config from (experiment path for non-MCQ, hint_type path for MCQ)

    Returns:
        Dict with "main" and "gepa_runs" keys where:
        - main: Candidate-level data (one row per candidate/is_sanitized) with metric values as columns.
            Overall metrics use column names like "proxy_reward".
            Subset-specific metrics use column names like "proxy_reward_hinted".
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

            # Get instructions for this candidate
            original_instructions = prog_point.get("original_instructions")
            sanitized_instructions = prog_point.get("sanitized_instructions")

            # Group overall scores by is_sanitized
            overall_by_sanitized = _group_scores_by_sanitized(prog_point["test_scores"])

            # Group subset scores by is_sanitized, with subset suffix in column names
            subset_scores_by_sanitized = defaultdict(dict)
            for subset_name, subset_scores in prog_point.get(
                "subset_test_scores", {}
            ).items():
                for is_sanitized, metric_scores in _group_scores_by_sanitized(
                    subset_scores
                ).items():
                    # Add subset suffix to each metric name
                    for metric_name, value in metric_scores.items():
                        subset_scores_by_sanitized[is_sanitized][
                            f"{metric_name}_{subset_name}"
                        ] = value

            # Create one row per is_sanitized value
            for is_sanitized in set(overall_by_sanitized.keys()) | set(
                subset_scores_by_sanitized.keys()
            ):
                instructions = (
                    sanitized_instructions if is_sanitized else original_instructions
                )
                rows["main"].append(
                    {
                        **base_row,
                        "is_sanitized": is_sanitized,
                        "instructions": instructions,
                        **overall_by_sanitized[is_sanitized],
                        **subset_scores_by_sanitized[is_sanitized],
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


def _group_scores_by_sanitized(scores: dict) -> dict[bool, dict[str, any]]:
    """Group scores by is_sanitized and return metric columns for each.

    Args:
        scores: Dict like {"proxy_reward_original": 0.85, "true_reward_sanitized": 0.72}

    Returns:
        Dict like {
            False: {"proxy_reward": 0.85, "true_reward": 0.42, "prompt_verbalizes": "yes"},
            True: {"proxy_reward": 0.80, "true_reward": 0.40}
        }
    """
    grouped = defaultdict(dict)
    for score_key, score_value in scores.items():
        metric_name, is_sanitized = _parse_score_key(score_key)
        grouped[is_sanitized][metric_name] = score_value
    return dict(grouped)


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


# =============================================================================
# Shortening DataFrame Loader
# =============================================================================


def load_shortening_dfs(paths: list[str]) -> DFs:
    """Load shortening results into DataFrame with one row per (run, threshold).

    Args:
        paths: List of paths to shortening experiment directories. Can be:
            - Timestamp level: "logs/shortening/wordchain/exp1/2025-01-15/" (loads all runs)
            - Specific run: "logs/shortening/wordchain/exp1/2025-01-15/baseline=.../0/"

    Returns:
        Dict with DataFrames:
        - "main": One row per (experiment, run, threshold) with scores and lengths
        - "candidates": One row per (experiment, run, candidate) with all candidate info
    """
    expanded_paths = _expand_shortening_paths(paths)

    all_main_rows = []
    all_candidate_rows = []

    for exp_path in expanded_paths:
        try:
            results = load_shortening_results(str(exp_path))
            metadata = _extract_shortening_metadata(exp_path)

            # Add rows for best_per_threshold
            for entry in results.get("best_per_threshold", []):
                all_main_rows.append(
                    {
                        **metadata,
                        "threshold": entry["threshold"],
                        "instructions": entry["instructions"],
                        "prompt_length": entry["prompt_length"],
                        "train_score": entry["train_score"],
                        "validation_score": entry["validation_score"],
                        "test_score": entry["test_score"],
                        "initial_prompt_path": results.get("initial_prompt_path"),
                        "initial_train_score": results.get("initial_train_score"),
                        "initial_length": results.get("initial_length"),
                    }
                )

            # Add rows for all candidates
            for candidate in results.get("all_candidates", []):
                is_pareto = candidate in results.get("pareto_optimal_candidates", [])
                all_candidate_rows.append(
                    {
                        **metadata,
                        "instructions": candidate["instructions"],
                        "prompt_length": candidate["prompt_length"],
                        "train_score": candidate["train_score"],
                        "iteration": candidate.get("iteration"),
                        "source": candidate.get("source"),
                        "is_pareto_optimal": is_pareto,
                    }
                )

        except FileNotFoundError:
            print(f"Warning: No shortening_results.json found in {exp_path}")
            continue

    if not all_main_rows:
        raise ValueError("No shortening data found in the specified paths")

    return {
        "main": pd.DataFrame(all_main_rows),
        "candidates": pd.DataFrame(all_candidate_rows),
    }


def _expand_shortening_paths(paths: list[str]) -> list[Path]:
    """Expand shortening paths to specific run directories.

    Looks for directories containing shortening_results.json.
    """
    expanded = []
    for path_str in [p for p in paths if p is not None]:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path does not exist: {path}")
            continue

        # Check if this path directly contains shortening_results.json
        if (path / "shortening_results.json").exists():
            expanded.append(path)
        else:
            # Search for shortening_results.json in subdirectories
            for results_file in path.rglob("shortening_results.json"):
                expanded.append(results_file.parent)

    return expanded


def _extract_shortening_metadata(exp_path: Path) -> dict:
    """Extract metadata from shortening experiment path."""
    metadata = {"experiment_path": str(exp_path)}

    # Try to load config.json for additional metadata
    config_path = exp_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        for key, value in config.items():
            metadata[key] = tuple(value) if isinstance(value, list) else value

    # Infer env_name from path if not in config
    if "env_name" not in metadata:
        path_str = str(exp_path)
        for env in KNOWN_ENVIRONMENTS:
            if env in path_str:
                metadata["env_name"] = env
                break

    return metadata
