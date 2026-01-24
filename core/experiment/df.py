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
from core.metrics import METRICS_BY_PROMPT_VERSION, METRICS
from core.shortening.results import load_shortening_results


# Type alias for the dictionary of DataFrames
DFs = dict[str, pd.DataFrame]


def load_dfs(
    paths: list[str],
    quick_mode: bool = False,
    metrics: list[str] | None = None,
) -> DFs:
    """Load experiment data from multiple paths into DataFrames.

    Auto-detects whether each experiment is GEPA or shortening data.

    Args:
        paths: List of paths to experiment directories. Can be:
            - Timestamp level: "logs/mcq/2025-11-19-11-41-43/" (loads all experiments under it)
            - Specific: "logs/mcq/2025-11-19-11-41-43/p=claude-..." (loads just that one)
        quick_mode: For GEPA only - if True, only evaluate first+last candidates
        metrics: For GEPA only - list of metric names to compute

    Returns:
        Dict with "main" DataFrame (and "gepa_runs" for GEPA data)
    """
    # Filter out None paths
    paths = [p for p in paths if p is not None]
    if not paths:
        raise ValueError("No paths provided")

    # Expand paths to leaf experiment directories
    expanded_paths = _expand_all_paths(paths)

    all_rows = defaultdict(list)
    for exp_path in expanded_paths:
        rows = _load_single_experiment(exp_path, quick_mode, metrics)
        for key in rows:
            all_rows[key].extend(rows[key])

    if not all_rows.get("main"):
        raise ValueError("No data found in the specified paths")

    return {key: pd.DataFrame(rows) for key, rows in all_rows.items()}


def _expand_all_paths(paths: list[str]) -> list[Path]:
    """Expand paths to leaf experiment directories.

    Handles both GEPA and shortening experiment structures.

    Args:
        paths: List of paths to experiment directories or parent directories

    Returns:
        List of specific experiment directory paths
    """
    expanded = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path does not exist: {path}")
            continue
        expanded.extend(_find_experiment_dirs(path))
    return expanded


def _find_experiment_dirs(path: Path) -> list[Path]:
    """Find all experiment directories at or under a path."""
    data_type = _get_data_type(path)
    if data_type is not None:
        return [path]

    # Recurse into subdirectories
    results = []
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir():
            results.extend(_find_experiment_dirs(subdir))
    return results


def _get_data_type(path: Path) -> str | None:
    """Detect the type of experiment data at a path, or None if not an experiment dir.

    Shortening: shortening_results.json in experiment dir
    GEPA: progression_data.json in run subdirs (0/, 1/, etc.)
    """
    data_types = []
    if (path / "shortening_results.json").exists():
        data_types.append("shortening")
    if list(path.glob("*/progression_data.json")):
        data_types.append("gepa")
    if len(data_types) > 1:
        raise ValueError(f"Ambiguous data types in {path}: {data_types}")
    return data_types[0] if data_types else None


def _load_single_experiment(
    exp_path: Path, quick_mode: bool, metrics: list[str] | None = None
) -> dict[str, list[dict]]:
    """Load data for a single experiment path into rows.

    Automatically detects whether the experiment is GEPA or shortening
    and calls the appropriate loader.

    Args:
        exp_path: Path to specific experiment directory
        quick_mode: For GEPA - whether to use quick mode for loading
        metrics: For GEPA - list of metric names to compute. If None, computes all.

    Returns:
        Dict with "main" key containing row lists (and "gepa_runs" for GEPA data)
    """
    data_type = _get_data_type(exp_path)
    assert data_type is not None, f"Not an experiment directory: {exp_path}"

    if data_type == "shortening":
        shortening_data = load_shortening_results(str(exp_path))
        return _flatten_shortening(shortening_data, exp_path)
    else:
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
    return {"experiment_path": str(exp_path), **config}


def _hashable_row(row: dict) -> dict:
    """Convert lists to tuples for DataFrame hashability."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}


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
                rows["main"].append(_hashable_row({
                    **base_row,
                    "is_sanitized": is_sanitized,
                    "instructions": instructions,
                    **overall_by_sanitized[is_sanitized],
                    **subset_scores_by_sanitized[is_sanitized],
                }))

        # Add run-level row to gepa_runs DataFrame
        end_state = run_data["end_state"]
        rows["gepa_runs"].append(_hashable_row({
            **metadata,
            "run_index": run_index,
            "end_discovery_eval_counts": end_state["discovery_eval_counts"],
            "end_reflection_call_count": end_state["reflection_call_count"],
        }))

    return rows


def _group_scores_by_sanitized(scores: dict) -> dict[bool, dict[str, any]]:
    """Group scores by is_sanitized and return metric columns for each.

    Args:
        scores: Dict like {"proxy_reward_original": 0.85, "true_reward_sanitized": 0.72}

    Returns:
        Dict like {
            False: {"proxy_reward": 0.85, "true_reward": 0.42, "verbalizes_prompt_yes": 0.6},
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


def _flatten_shortening(shortening_data: dict, exp_path: Path) -> dict[str, list[dict]]:
    """Flatten shortening data to rows."""
    config_path = exp_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    metadata = _extract_metadata(exp_path, config)

    pareto_instructions = {
        entry["instructions"]
        for entry in shortening_data.get("pareto_frontier_results", [])
    }

    rows = []
    for candidate in shortening_data["all_candidates"]:
        rows.append(_hashable_row({
            **metadata,
            "instructions": candidate["instructions"],
            "prompt_length": candidate["prompt_length"],
            "val_score": candidate["val_score"],
            "test_score": candidate.get("test_score"),
            "iteration": candidate["iteration"],
            "source": candidate["source"],
            "pieces_removed": candidate.get("pieces_removed"),
            "is_pareto": candidate["instructions"] in pareto_instructions,
            "initial_prompt_path": shortening_data["initial_prompt_path"],
        }))

    return {"main": rows}
