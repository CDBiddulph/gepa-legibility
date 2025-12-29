"""
Progression data collection for GEPA experiments.

This module provides utilities to load or compute progression data for GEPA experiments,
with automatic caching support.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from core.evaluation.loaders import EvalData, load_eval_data
from core.experiment_utils import (
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


def get_progression_data(experiment_path, quick_mode=False):
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

    Returns:
        {'runs': [...]}
    """
    experiment_path = Path(experiment_path)
    env = get_environment_from_path(experiment_path)

    # Get executor model and configure DSPy
    executor_model_name = get_executor_model_name_from_path(experiment_path)
    print(f"Using executor: {executor_model_name}")
    executor_lm = get_dspy_lm(executor_model_name, cache=True)

    # Load evaluation data
    eval_data = load_eval_data(env, split="test")

    # Collect progression data
    return _collect_progression_data(
        experiment_path,
        eval_data,
        executor_lm,
        quick_mode,
    )


# ============================================================================
# Private Helper Functions
# ============================================================================


def _collect_progression_data(
    experiment_path,
    eval_data: EvalData,
    executor_lm,
    quick_mode=False,
):
    """
    Collect progression data for a single experiment path.

    Args:
        experiment_path: Path to experiment directory containing run subdirectories
        eval_data: Loaded evaluation data
        executor_lm: Language model for evaluation
        quick_mode: If True, only evaluate first+last candidates (for bar plots)

    Returns:
        {'runs': [...]}
    """
    # Get hint_type from config.json (will be None for non-MCQ experiments)
    hint_type = _get_hint_type_from_config(experiment_path)

    print(f"Collecting progression data for {experiment_path}")
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
            hint_type,
            use_incompetent_adapter,
            quick_mode=quick_mode,
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


def _get_hint_type_from_config(experiment_path):
    """Get hint_type from config.json in first run directory, or None if not present."""
    config_files = sorted(experiment_path.glob("*/config.json"))
    if not config_files:
        return None
    with open(config_files[0]) as f:
        config = json.load(f)
    return config.get("hint_type")


def _process_run_progression(
    data,
    subdir,
    eval_data: EvalData,
    executor_lm,
    hint_type: str | None,
    use_incompetent_adapter: bool,
    quick_mode=False,
):
    """
    Process progression data for a single run.

    Args:
        data: Loaded detailed_results.json data
        subdir: Path to run subdirectory
        eval_data: Loaded evaluation data
        executor_lm: Language model for evaluation
        hint_type: For MCQ - which hint type (None for non-MCQ)
        use_incompetent_adapter: Whether to use IncompetentAdapter
        quick_mode: If True, only evaluate first and last candidates

    Returns:
        List of progression point dicts
    """
    # First pass: find improving candidates and best index
    best_val_score = -float("inf")
    best_idx = 0
    improving_candidates = []

    for idx, candidate in enumerate(data["candidates"]):
        val_score = candidate["val_aggregate_score"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_idx = idx
            improving_candidates.append(idx)

    # Determine which candidates to evaluate based on mode
    if quick_mode:
        # Quick mode: only first and last
        candidates_to_eval = {0, best_idx}
    else:
        # Full mode: all improving candidates
        candidates_to_eval = set(improving_candidates)

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
            hint_type,
            use_incompetent_adapter,
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
    eval_data: EvalData,
    executor_lm,
    hint_type: str | None,
    use_incompetent_adapter: bool,
):
    """
    Evaluate a single candidate on all metrics.

    Uses the metrics registry to determine which metrics to compute for each
    prompt version (original/sanitized). Results are cached in evaluations/.

    Returns:
        Tuple of (test_scores, subset_test_scores, sanitized_instructions)
    """
    test_scores = {}
    subset_test_scores = defaultdict(dict)
    eval_dir = subdir / "evaluations"

    # Determine which versions to evaluate
    # All candidates (including baseline) use real instructions from detailed_results.json
    original_instructions = candidate["instructions"]
    sanitized_instructions = load_instructions(subdir, idx, env=eval_data.env, sanitized=True)
    versions = {
        "original": original_instructions,
        "sanitized": sanitized_instructions,
    }

    # Select adapter
    adapter = "incompetent" if use_incompetent_adapter else "chat"

    # Evaluate each version with its applicable metrics
    for version_name, instructions in versions.items():
        # Get metrics for this version
        metrics_to_eval = METRICS_BY_PROMPT_VERSION[version_name]

        for metric_name in metrics_to_eval:
            cache_file = eval_dir / f"cand_{idx}_{metric_name}_{version_name}.json"

            # Try to load from cache first
            if cache_file.exists():
                result = _load_cached_metric(cache_file, metric_name)
            else:
                # Create context and call metric's calculate method
                context = MetricContext(
                    instructions=instructions,
                    prompt_version=version_name,
                    hint_type=hint_type,
                    eval_data=eval_data,
                    executor_lm=executor_lm,
                    adapter=adapter,
                )
                result, cache_data = METRICS[metric_name].calculate(context)

                # Save to cache
                _save_cached_metric(cache_file, cache_data)

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


def _load_cached_metric(cache_file: Path, metric_name: str) -> MetricResult:
    """Load metric result from cache file using the metric's from_cache method."""
    with open(cache_file) as f:
        cache_data = json.load(f)

    return METRICS[metric_name].from_cache(cache_data)


def _save_cached_metric(cache_file: Path, cache_data: dict):
    """Save metric cache data to file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


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
        help="Path to experiment directory (e.g., logs/psychosis/.../p=...-hack=no)",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Only evaluate first and last candidates (for bar plots)",
    )

    args = parser.parse_args()

    # Load progression data
    print(f"Loading progression data for: {args.experiment_path}")
    data = get_progression_data(args.experiment_path, quick_mode=args.quick_mode)

    # Print summary
    print(f"\nLoaded progression data with {len(data['runs'])} runs")

    print("\nDone!")


if __name__ == "__main__":
    main()
