"""
Progression data collection for GEPA experiments.

This module provides utilities to load or compute progression data for experiments,
with automatic caching support. It handles both:
- GEPA experiments: Multiple candidates with optimized instructions
- Baseline-only experiments: Single candidate with no optimization (prompter_name=None)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from core.evaluation.loaders import EvalData, load_eval_data
from core.experiment_utils import (
    expand_experiment_paths,
    get_environment_from_path,
    get_executor_model_name_from_path,
    uses_incompetent_adapter,
)
from core.instruction_loader import load_instructions
from core.lm import get_dspy_lm
from core.metrics import (
    METRICS,
    METRICS_BY_PROMPT_VERSION,
    MetricContext,
    MetricResult,
)

# Load environment variables
load_dotenv()


# ============================================================================
# Public API
# ============================================================================


def get_progression_data(experiment_path, quick_mode=False, metrics=None):
    """
    Compute progression data for a single experiment path.

    This function builds progression data by evaluating candidates and aggregating
    results. Individual evaluation results are cached in evaluations/*.json files,
    so repeated calls are fast. The progression_data.json file is regenerated on
    each call for human readability but is not used as a cache.

    Args:
        experiment_path: Path to experiment directory (str or Path)
        quick_mode: If True, only evaluate first+last candidates (for bar plots).
                   If False, evaluate all improving candidates (for progression plots).
        metrics: List of metric names to compute. If None, computes all metrics.

    Returns:
        {'runs': [...]}
    """
    experiment_path = Path(experiment_path)
    env = get_environment_from_path(experiment_path)

    # Check if all metrics are cached - skip expensive loading if so
    if _all_metrics_cached(experiment_path, quick_mode, metrics):
        print(f"All metrics cached for {experiment_path.name}")
        eval_data, executor_lm = None, None
    else:
        executor_model_name = get_executor_model_name_from_path(experiment_path)
        print(f"Using executor: {executor_model_name}")
        executor_lm = get_dspy_lm(executor_model_name, cache=True)
        eval_data = load_eval_data(env, split="test")

    return _collect_progression_data(
        experiment_path, eval_data, executor_lm, env, quick_mode, metrics
    )


# ============================================================================
# Private Helper Functions
# ============================================================================


def _get_candidates_to_eval(data: dict, quick_mode: bool) -> set[int]:
    """Determine which candidate indices to evaluate based on mode."""
    best_val_score = -float("inf")
    best_idx = 0
    improving_candidates = []

    for idx, candidate in enumerate(data["candidates"]):
        val_score = candidate["val_aggregate_score"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_idx = idx
            improving_candidates.append(idx)

    if quick_mode:
        return {0, best_idx}
    else:
        return set(improving_candidates)


def _get_required_cache_files(
    subdir: Path, candidate_indices: set[int], metrics: list[str] | None
) -> list[Path]:
    """Get list of cache files required for given candidates and metrics."""
    eval_dir = subdir / "evaluations"
    cache_files = []

    for idx in candidate_indices:
        for version_name in ["original", "sanitized"]:
            all_metrics_for_version = METRICS_BY_PROMPT_VERSION[version_name]
            if metrics is not None:
                metrics_to_check = [m for m in all_metrics_for_version if m in metrics]
            else:
                metrics_to_check = all_metrics_for_version

            for metric_name in metrics_to_check:
                cache_files.append(eval_dir / f"cand_{idx}_{metric_name}_{version_name}.json")

    return cache_files


def _all_metrics_cached(experiment_path: Path, quick_mode: bool, metrics: list[str] | None) -> bool:
    """Check if all required metrics are cached for an experiment."""
    for subdir in experiment_path.iterdir():
        if not subdir.is_dir():
            continue

        results_file = subdir / "detailed_results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        candidates_to_eval = _get_candidates_to_eval(data, quick_mode)
        required_files = _get_required_cache_files(subdir, candidates_to_eval, metrics)

        for cache_file in required_files:
            if not cache_file.exists():
                return False

    return True


def _collect_progression_data(
    experiment_path,
    eval_data: EvalData | None,
    executor_lm,
    env: str,
    quick_mode=False,
    metrics=None,
):
    """
    Collect progression data for a single experiment path.

    Args:
        experiment_path: Path to experiment directory containing run subdirectories
        eval_data: Loaded evaluation data (None if all metrics cached)
        executor_lm: Language model for evaluation (None if all metrics cached)
        env: Environment name (e.g., "mcq", "psychosis")
        quick_mode: If True, only evaluate first+last candidates (for bar plots)
        metrics: List of metric names to compute. If None, computes all metrics.

    Returns:
        {'runs': [...]}
    """
    # Get hint_type from config.json (will be None for non-MCQ experiments)
    hint_type = _get_hint_type_from_config(experiment_path)

    # Check if this is a baseline-only experiment (no optimization)
    is_baseline_only = _is_baseline_only(experiment_path)

    print(f"Collecting progression data for {experiment_path}")
    if is_baseline_only:
        print(f"  Baseline-only experiment detected")

    # Check if this experiment uses IncompetentAdapter
    use_incompetent_adapter = uses_incompetent_adapter(experiment_path)
    if use_incompetent_adapter:
        print(
            f"  Detected 'incompetent' in path - will use IncompetentAdapter during evaluation"
        )

    if quick_mode:
        print(f"  Quick mode enabled: only evaluating first and final candidates")

    runs_data = {}

    # Iterate through each run subdirectory
    for subdir in sorted(experiment_path.iterdir()):
        if not subdir.is_dir():
            continue

        results_file = subdir / "detailed_results.json"
        if not results_file.exists():
            continue

        run_index = int(subdir.name)
        run_file = subdir / "progression_data.json"

        with open(results_file, "r") as f:
            data = json.load(f)

        # Process progression for this run
        progression = _process_run_progression(
            data,
            subdir,
            eval_data,
            executor_lm,
            env,
            hint_type,
            use_incompetent_adapter,
            is_baseline_only=is_baseline_only,
            quick_mode=quick_mode,
            metrics=metrics,
        )

        # Construct data for the end state
        end_state = {
            "candidate_index": len(data["candidates"]) - 1,
            "discovery_eval_counts": data["candidates"][-1]["discovery_eval_counts"],
            "reflection_call_count": data["candidates"][-1]["reflection_call_count"],
        }

        run_data = {
            "run_index": run_index,
            "progression": progression,
            "end_state": end_state,
        }
        runs_data[run_index] = run_data

        # Save progression data in run directory
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2)
        logger.info(f"Saved progression data to {run_file}")

    # Return runs sorted by run_index
    return {"runs": [runs_data[k] for k in sorted(runs_data.keys())]}


def _get_config_from_experiment(experiment_path):
    """Get config.json from first run directory, or empty dict if not present."""
    config_files = sorted(experiment_path.glob("*/config.json"))
    if not config_files:
        return {}
    with open(config_files[0]) as f:
        return json.load(f)


def _get_hint_type_from_config(experiment_path):
    """Get hint_type from config.json in first run directory, or None if not present."""
    config = _get_config_from_experiment(experiment_path)
    return config.get("hint_type")


def _is_baseline_only(experiment_path):
    """Check if experiment is baseline-only (no GEPA optimization)."""
    config = _get_config_from_experiment(experiment_path)
    return config.get("prompter_name") is None


def _process_run_progression(
    data,
    subdir,
    eval_data: EvalData | None,
    executor_lm,
    env: str,
    hint_type: str | None,
    use_incompetent_adapter: bool,
    is_baseline_only: bool = False,
    quick_mode=False,
    metrics=None,
):
    """
    Process progression data for a single run.

    Args:
        data: Loaded detailed_results.json data
        subdir: Path to run subdirectory
        eval_data: Loaded evaluation data (None if all metrics cached)
        executor_lm: Language model for evaluation (None if all metrics cached)
        env: Environment name
        hint_type: For MCQ - which hint type (None for non-MCQ)
        use_incompetent_adapter: Whether to use IncompetentAdapter
        is_baseline_only: Whether this is a baseline-only experiment (no optimization)
        quick_mode: If True, only evaluate first and last candidates
        metrics: List of metric names to compute. If None, computes all metrics.

    Returns:
        List of progression point dicts
    """
    candidates_to_eval = _get_candidates_to_eval(data, quick_mode)

    # Evaluate selected candidates
    progression = []
    for idx in sorted(candidates_to_eval):
        candidate = data["candidates"][idx]
        val_score = candidate["val_aggregate_score"]

        logger.info(f"Run {subdir.name}, candidate {idx} (val={val_score:.3f})")

        # Evaluate candidate with detailed capture
        test_scores, subset_test_scores, sanitized_instructions = _evaluate_candidate(
            candidate,
            idx,
            subdir,
            eval_data,
            executor_lm,
            env,
            hint_type,
            use_incompetent_adapter,
            is_baseline_only=is_baseline_only,
            metrics=metrics,
        )

        prog_point = _create_progression_point(
            candidate, idx, test_scores, subset_test_scores, sanitized_instructions
        )
        progression.append(prog_point)

    return progression


def _evaluate_candidate(
    candidate,
    idx,
    subdir,
    eval_data: EvalData | None,
    executor_lm,
    env: str,
    hint_type: str | None,
    use_incompetent_adapter: bool,
    is_baseline_only: bool = False,
    metrics=None,
):
    """
    Evaluate a single candidate on specified metrics.

    Uses the metrics registry to determine which metrics to compute for each
    prompt version (original/sanitized). Results are cached in evaluations/.
    eval_data and executor_lm can be None if all metrics are cached.

    For baseline-only experiments:
    - Only evaluates "original" version (no sanitized - nothing to sanitize)

    Returns:
        Tuple of (test_scores, subset_test_scores, sanitized_instructions)
    """
    # Since eval_data may be None, we include env in the function signature.
    # Check that if eval_data is *not* None, it is consistent with env.
    if eval_data is not None:
        assert eval_data.env == env

    test_scores = {}
    subset_test_scores = defaultdict(dict)
    eval_dir = subdir / "evaluations"

    # Determine which versions to evaluate
    original_instructions = candidate["instructions"]

    if is_baseline_only:
        # Baseline-only: only "original" version, no sanitization
        versions = {"original": original_instructions}
        sanitized_instructions = None
    else:
        # GEPA: both original and sanitized versions
        sanitized_instructions = load_instructions(subdir, idx, env=env, sanitized=True)
        versions = {
            "original": original_instructions,
            "sanitized": sanitized_instructions,
        }

    # Select adapter
    adapter = "incompetent" if use_incompetent_adapter else "chat"

    # Evaluate each version with its applicable metrics
    for version_name, instructions in versions.items():
        # Get metrics for this version, filtered by requested metrics if specified
        all_metrics_for_version = METRICS_BY_PROMPT_VERSION[version_name]

        if metrics is not None:
            metrics_to_eval = [m for m in all_metrics_for_version if m in metrics]
        else:
            metrics_to_eval = all_metrics_for_version

        for metric_name in metrics_to_eval:
            cache_file = eval_dir / f"cand_{idx}_{metric_name}_{version_name}.json"

            # Create context and evaluate with caching
            context = MetricContext(
                instructions=instructions,
                prompt_version=version_name,
                hint_type=hint_type,
                eval_data=eval_data,
                executor_lm=executor_lm,
                adapter=adapter,
            )
            result = evaluate_metric_cached(metric_name, context, cache_file)

            # Store result
            score_key = f"{metric_name}_{version_name}"
            test_scores[score_key] = result.value

            # Store subset scores if present
            if result.subset_values:
                for subset_name, subset_score in result.subset_values.items():
                    subset_test_scores[subset_name][score_key] = subset_score

    return test_scores, dict(subset_test_scores), sanitized_instructions


def _create_progression_point(
    candidate, idx, test_scores, subset_test_scores, sanitized_instructions
):
    """
    Create a progression point dict with proper field ordering.

    Args:
        candidate: Candidate data from detailed_results.json
        idx: Candidate index
        test_scores: Dict of test scores (e.g., proxy_reward_original, true_reward_sanitized)
        subset_test_scores: Dict of subset -> scores, or empty dict if no subsets
        sanitized_instructions: Sanitized instructions string or None

    Returns:
        Dict representing a progression point
    """
    prog_point = {
        "candidate_index": idx,
        "validation_score": candidate["val_aggregate_score"],
        "discovery_eval_counts": candidate["discovery_eval_counts"],
        "reflection_call_count": candidate["reflection_call_count"],
        # TODO: rename to "metrics"
        "test_scores": test_scores,
    }

    # Add subset scores immediately after test_scores if they exist
    if subset_test_scores:
        prog_point["subset_test_scores"] = subset_test_scores

    prog_point["original_instructions"] = candidate["instructions"]
    prog_point["sanitized_instructions"] = sanitized_instructions

    return prog_point


def load_cached_metric(cache_file: Path, metric_name: str) -> MetricResult:
    """Load metric result from cache file using the metric's from_cache method."""
    with open(cache_file) as f:
        cache_data = json.load(f)

    return METRICS[metric_name].from_cache(cache_data)


def save_cached_metric(cache_file: Path, cache_data: dict):
    """Save metric cache data to file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def evaluate_metric_cached(
    metric_name: str,
    context: MetricContext,
    cache_file: Path,
) -> MetricResult:
    """
    Evaluate a metric with caching support.

    This is the core caching logic used by both GEPA candidate evaluation
    and RL model evaluation.

    Args:
        metric_name: Name of the metric to evaluate
        context: MetricContext with instructions, eval_data, executor_lm, etc.
        cache_file: Path to cache file

    Returns:
        MetricResult with value and optional subset_values
    """
    if cache_file.exists():
        return load_cached_metric(cache_file, metric_name)

    result, cache_data = METRICS[metric_name].calculate(context)
    save_cached_metric(cache_file, cache_data)
    return result


# ============================================================================
# CLI Main Function
# ============================================================================


def main():
    """CLI entry point for generating progression data."""
    import argparse

    # Configure logging to show INFO level for CLI usage
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Load or generate progression data for GEPA experiments"
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to experiment directory or timestamp directory. "
        "If a timestamp directory (e.g., logs/mcq/.../2025-12-23-00-02-21/), "
        "processes all subdirectories.",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Only evaluate first and last candidates (for bar plots)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Only compute these metrics (e.g., --metrics proxy_reward true_reward)",
    )

    args = parser.parse_args()

    # Expand path to list of experiment directories
    input_path = Path(args.experiment_path)
    experiment_paths = expand_experiment_paths(input_path)

    if len(experiment_paths) > 1:
        print(f"Found {len(experiment_paths)} experiment directories under {input_path}")

    total_runs = 0
    for i, exp_path in enumerate(experiment_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(experiment_paths)}] Processing: {exp_path.name}")
        print('='*60)

        data = get_progression_data(
            exp_path, quick_mode=args.quick_mode, metrics=args.metrics
        )
        total_runs += len(data['runs'])
        print(f"  Loaded {len(data['runs'])} runs")

    # Print summary
    print(f"\nDone! Processed {len(experiment_paths)} experiments with {total_runs} total runs.")


if __name__ == "__main__":
    main()
