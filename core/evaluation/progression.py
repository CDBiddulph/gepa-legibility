"""
Progression data collection for GEPA experiments.

This module provides utilities to load or compute progression data for experiments,
with automatic caching support. It handles both:
- GEPA experiments: Multiple candidates with optimized instructions
- Baseline-only experiments: Single candidate with no optimization (prompter_name=None)
"""

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from core.evaluation.loaders import EvalData, load_eval_data
from core.experiment_utils import (
    expand_experiment_paths,
    get_environment_from_path,
    get_executor_model_name_from_path,
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


def get_progression_data(experiment_path, quick_mode=False, metrics=None, num_threads=32):
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
        num_threads: Number of threads for parallel evaluation (default: 32).

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
        experiment_path, eval_data, executor_lm, env, quick_mode, metrics, num_threads
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


def _cache_exists_for_metric(
    eval_dir: Path, candidate_idx: int, metric_name: str, version_name: str
) -> bool:
    """Check if cache exists for a metric, including fallback.

    Checks the primary cache file first, then the fallback file for
    metrics with shared caches.
    """
    cache_file = get_cache_file_for_metric(eval_dir, candidate_idx, metric_name, version_name)
    if cache_file.exists():
        return True

    fallback_file = _get_fallback_cache_file(eval_dir, candidate_idx, metric_name, version_name)
    if fallback_file and fallback_file.exists():
        return True

    return False


def _all_metrics_cached(experiment_path: Path, quick_mode: bool, metrics: list[str] | None) -> bool:
    """Check if all required metrics are cached for an experiment.

    Supports shared caches: metrics with the same cache_key share a cache file.
    Also checks fallback files for backward compatibility.
    """
    # Check if this is a baseline-only experiment
    config = _get_config_from_experiment(experiment_path)
    is_baseline_only = config.get("prompter_name") is None

    # Baseline-only experiments only have "original" version (no sanitization)
    versions_to_check = ["original"] if is_baseline_only else ["original", "sanitized"]

    for subdir in experiment_path.iterdir():
        if not subdir.is_dir():
            continue

        results_file = subdir / "detailed_results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        eval_dir = subdir / "evaluations"
        candidates_to_eval = _get_candidates_to_eval(data, quick_mode)

        for idx in candidates_to_eval:
            for version_name in versions_to_check:
                all_metrics_for_version = METRICS_BY_PROMPT_VERSION[version_name]
                if metrics is not None:
                    metrics_to_check = [m for m in all_metrics_for_version if m in metrics]
                else:
                    metrics_to_check = all_metrics_for_version

                # Deduplicate by cache_key to avoid redundant checks
                checked_cache_keys = set()
                for metric_name in metrics_to_check:
                    metric_cls = METRICS[metric_name]
                    cache_key = getattr(metric_cls, "cache_key", None) or metric_name

                    if cache_key in checked_cache_keys:
                        continue
                    checked_cache_keys.add(cache_key)

                    if not _cache_exists_for_metric(eval_dir, idx, metric_name, version_name):
                        return False

    return True


def _collect_progression_data(
    experiment_path,
    eval_data: EvalData | None,
    executor_lm,
    env: str,
    quick_mode=False,
    metrics=None,
    num_threads=32,
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
        num_threads: Number of threads for parallel evaluation.

    Returns:
        {'runs': [...]}
    """
    # Load config and extract values
    config = _get_config_from_experiment(experiment_path)
    hint_type = config.get("hint_type")  # None for non-MCQ experiments
    is_baseline_only = config.get("prompter_name") is None
    use_incompetent_adapter = config["incompetent"]

    print(f"Collecting progression data for {experiment_path}")
    if is_baseline_only:
        print("  Baseline-only experiment detected")
    if use_incompetent_adapter:
        print("  Using IncompetentAdapter during evaluation")

    if quick_mode:
        print("  Quick mode enabled: only evaluating first and final candidates")

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
            num_threads=num_threads,
        )

        # Construct data for the end state
        end_state = {
            "candidate_index": len(data["candidates"]) - 1,
            # Baseline-only experiments don't have total_metric_calls, so add a fallback
            "discovery_eval_counts": data.get("total_metric_calls", data["candidates"][-1]["discovery_eval_counts"]),
            # This is technically not quite correct, because it doesn't include failed reflections
            # after the last candidate. The lines won't extend all the way to the right.
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
    num_threads=32,
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
        num_threads: Number of threads for parallel evaluation.

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
            num_threads=num_threads,
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
    num_threads=32,
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
        sanitized_instructions = load_instructions(subdir, idx, env=env, sanitized=True, hint_type=hint_type)
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
            # Create context and evaluate with caching
            context = MetricContext(
                instructions=instructions,
                prompt_version=version_name,
                hint_type=hint_type,
                eval_data=eval_data,
                executor_lm=executor_lm,
                adapter=adapter,
                num_threads=num_threads,
            )
            result = evaluate_metric_cached(metric_name, context, eval_dir, idx, version_name)

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


def _get_metric_cache_key(metric_name: str) -> str:
    """Get the cache key for a metric (shared key if set, otherwise metric name)."""
    metric_cls = METRICS[metric_name]
    return getattr(metric_cls, "cache_key", None) or metric_name


def _get_fallback_cache_file(
    eval_dir: Path, candidate_idx: int, metric_name: str, version_name: str
) -> Path | None:
    """Get fallback cache file path for backward compatibility, if applicable.

    Returns the old-style cache file path (e.g., verbalizes_prompt_yes.json)
    that might exist from before the shared cache migration.
    """
    metric_cls = METRICS[metric_name]
    cache_key = getattr(metric_cls, "cache_key", None)
    fallback_suffix = getattr(metric_cls, "cache_fallback_suffix", None)

    if not cache_key or not fallback_suffix:
        return None

    return eval_dir / f"cand_{candidate_idx}_{cache_key}{fallback_suffix}_{version_name}.json"


def get_cache_file_for_metric(
    eval_dir: Path, candidate_idx: int, metric_name: str, version_name: str
) -> Path:
    """Get the primary cache file path for a metric.

    If the metric has a cache_key, uses that instead of the metric name.
    This allows multiple metrics to share the same cache file.
    """
    cache_key = _get_metric_cache_key(metric_name)
    return eval_dir / f"cand_{candidate_idx}_{cache_key}_{version_name}.json"


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
    eval_dir: Path,
    candidate_idx: int,
    version_name: str,
) -> MetricResult:
    """
    Evaluate a metric with caching support.

    Supports shared caches: if a metric has cache_key set, multiple metrics
    can share the same cache file. Also checks fallback file for backward
    compatibility with old individual cache files.

    Args:
        metric_name: Name of the metric to evaluate
        context: MetricContext with instructions, eval_data, executor_lm, etc.
        eval_dir: Directory containing cache files (e.g., run_dir/evaluations)
        candidate_idx: Candidate index
        version_name: "original" or "sanitized"

    Returns:
        MetricResult with value and optional subset_values
    """
    cache_file = get_cache_file_for_metric(eval_dir, candidate_idx, metric_name, version_name)

    # Check primary cache file
    if cache_file.exists():
        return load_cached_metric(cache_file, metric_name)

    # Check fallback file (old individual cache file)
    fallback_file = _get_fallback_cache_file(eval_dir, candidate_idx, metric_name, version_name)
    if fallback_file and fallback_file.exists():
        result = load_cached_metric(fallback_file, metric_name)
        # Copy to primary location for future use
        shutil.copy(fallback_file, cache_file)
        return result

    # Compute fresh and save to primary cache
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of threads for parallel evaluation (default: 32)",
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
            exp_path, quick_mode=args.quick_mode, metrics=args.metrics, num_threads=args.num_threads
        )
        total_runs += len(data['runs'])
        print(f"  Loaded {len(data['runs'])} runs")

    # Print summary
    print(f"\nDone! Processed {len(experiment_paths)} experiments with {total_runs} total runs.")


if __name__ == "__main__":
    main()
