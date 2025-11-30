"""
Progression data collection for GEPA experiments.

This module provides utilities to load or compute progression data for GEPA experiments,
with automatic caching support.
"""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from core.evaluation.evaluate import EvalResult, evaluate
from core.evaluation.loaders import EvalData, load_eval_data
from core.experiment_utils import (
    get_environment_from_path,
    get_executor_model_name_from_path,
    uses_incompetent_adapter,
)
from core.instruction_loader import load_instructions
from core.lm import get_dspy_lm

# Load environment variables
load_dotenv()


# ============================================================================
# Public API
# ============================================================================


def get_progression_data(experiment_path, quick_mode=False):
    """
    Load or compute progression data for a single experiment path.

    This function checks for cached progression data and loads it if available.
    If cache doesn't exist, it collects the data and caches it.

    Args:
        experiment_path: Path to experiment directory (str or Path)
        quick_mode: If True, only evaluate first+last candidates (for bar plots).
                   If False, evaluate all improving candidates (for progression plots).

    Returns:
        For non-MCQ environments: {'runs': [...]}
        For MCQ environment: {hint_type: {'runs': [...]}, ...}
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
        env,
        eval_data,
        executor_lm,
        quick_mode,
    )


# ============================================================================
# Private Helper Functions
# ============================================================================


def _collect_progression_data(
    experiment_path,
    env,
    eval_data: EvalData,
    executor_lm,
    quick_mode=False,
):
    """
    Collect progression data for a single experiment path.

    Args:
        experiment_path: Path to experiment directory
        env: Environment name
        eval_data: Loaded evaluation data
        executor_lm: Language model for evaluation
        quick_mode: If True, only evaluate first+last candidates

    Returns:
        For non-MCQ: {'runs': [...]}
        For MCQ: {hint_type: {'runs': [...]}, ...}
    """
    if env == "mcq":
        # MCQ: iterate through hint type subdirectories
        print(f"Collecting MCQ progression data for {experiment_path}")
        hint_data = {}

        for hint_dir in sorted(experiment_path.iterdir()):
            if not hint_dir.is_dir():
                continue

            hint_type = hint_dir.name
            print(f"  Collecting progression data for hint type: {hint_type}")

            hint_data[hint_type] = _collect_progression_data_generic(
                hint_dir,
                eval_data,
                executor_lm,
                env="mcq",
                hint_type=hint_type,
                quick_mode=quick_mode,
            )

        return hint_data
    else:
        # Non-MCQ: standard processing
        print(f"Collecting progression data for {experiment_path}")

        return _collect_progression_data_generic(
            experiment_path,
            eval_data,
            executor_lm,
            env=env,
            hint_type=None,
            quick_mode=quick_mode,
        )


def _collect_progression_data_generic(
    experiment_path,
    eval_data: EvalData,
    executor_lm,
    env: str,
    hint_type: str | None,
    quick_mode=False,
):
    """
    Generic progression data collection for a single experiment directory.

    Args:
        experiment_path: Path to experiment directory containing run subdirectories
        eval_data: Loaded evaluation data
        executor_lm: Language model for evaluation
        env: Environment name
        hint_type: For MCQ - which hint type (None for non-MCQ)
        quick_mode: If True, only evaluate first+last (for bar plots)

    Returns:
        Dict with 'runs' key containing list of run data
    """
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
        # Progression data now stored in the run directory itself
        run_file = subdir / "progression_data.json"

        # Load existing progression data if available
        existing_progression = []
        if run_file.exists():
            with open(run_file, "r") as f:
                existing_data = json.load(f)
                existing_progression = existing_data["progression"]

        with open(results_file, "r") as f:
            data = json.load(f)

        # Process progression for this run
        progression = _process_run_progression(
            data,
            existing_progression,
            subdir,
            eval_data,
            executor_lm,
            env,
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
        print(f"  Saved progression data to {run_file}")

    # Return runs sorted by run_index
    return {"runs": [runs_data[k] for k in sorted(runs_data.keys())]}


def _process_run_progression(
    data,
    existing_progression,
    subdir,
    eval_data: EvalData,
    executor_lm,
    env: str,
    hint_type: str | None,
    use_incompetent_adapter: bool,
    quick_mode=False,
):
    """
    Process progression data for a single run.

    Args:
        data: Loaded detailed_results.json data
        existing_progression: List of already-evaluated progression points
        subdir: Path to run subdirectory
        eval_data: Loaded evaluation data
        executor_lm: Language model for evaluation
        env: Environment name
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

    # Remove candidates that have already been evaluated
    candidates_to_eval = candidates_to_eval - {
        candidate["candidate_index"] for candidate in existing_progression
    }

    # Evaluate selected candidates
    progression = existing_progression
    for idx in sorted(candidates_to_eval):
        candidate = data["candidates"][idx]
        val_score = candidate["val_aggregate_score"]

        print(f"  Run {subdir.name}, candidate {idx} (val={val_score:.3f})")

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
        )

        prog_point = _create_progression_point(
            candidate, idx, test_scores, subset_test_scores, sanitized_instructions
        )
        progression.append(prog_point)

    # Sort progression by index
    progression.sort(key=lambda x: x["candidate_index"])

    return progression


def _evaluate_candidate(
    candidate,
    idx,
    subdir,
    eval_data: EvalData,
    executor_lm,
    env: str,
    hint_type: str | None,
    use_incompetent_adapter: bool,
):
    """
    Evaluate a single candidate on all test conditions with detailed capture.

    Returns:
        Tuple of (test_scores, subset_test_scores, sanitized_instructions)
    """
    test_scores = {}
    subset_test_scores = defaultdict(dict)
    eval_dir = subdir / "evaluations"

    # Determine which versions to evaluate
    if idx == 0:
        # Baseline: no instructions, same for both original and sanitized
        versions = [("original", None), ("sanitized", None)]
        sanitized_instructions = None
    else:
        # Non-baseline: evaluate both original and sanitized instructions
        original_instructions = candidate["instructions"]
        sanitized_instructions = load_instructions(subdir, idx, sanitized=True)
        versions = [
            ("original", original_instructions),
            ("sanitized", sanitized_instructions),
        ]

    # Select adapter
    adapter = "incompetent" if use_incompetent_adapter else "chat"

    # Evaluate all version × metric combinations
    for version_name, instructions in versions:
        for reward_type in ["proxy", "true"]:
            cache_file = eval_dir / f"cand_{idx}_{reward_type}_{version_name}.json"

            # Try to load from cache first
            if cache_file.exists():
                mean_score, subset_mean_scores = _load_cached_result(
                    cache_file, eval_data, hint_type
                )
            else:
                # Run evaluation
                result = evaluate(
                    eval_data,
                    executor_lm,
                    reward_type,
                    instructions=instructions,
                    hint_type=hint_type,
                    adapter=adapter,
                )

                mean_score = result.score
                subset_mean_scores = result.subset_scores

                # Save to cache
                _save_cached_result(cache_file, instructions, result)

            score_key = f"{reward_type}_{version_name}"
            test_scores[score_key] = mean_score

            # Collect subset scores
            for subset_name, subset_score in subset_mean_scores.items():
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
        test_scores: Dict of test scores (proxy_original, true_original, etc.)
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
        "test_scores": test_scores,
    }

    # Add subset scores immediately after test_scores if they exist
    if subset_test_scores:
        prog_point["subset_test_scores"] = subset_test_scores

    prog_point["original_instructions"] = candidate["instructions"]
    prog_point["sanitized_instructions"] = sanitized_instructions

    return prog_point


def _load_cached_result(
    cache_file: Path,
    eval_data: EvalData,
    hint_type: str | None,
) -> tuple[float, dict[str, float]]:
    """
    Load evaluation result from cache file.

    Returns:
        Tuple of (mean_score, subset_mean_scores)
    """
    with open(cache_file) as f:
        data = json.load(f)

    detailed_results = data["evaluations"]
    mean_score = sum(r["score"] for r in detailed_results) / len(detailed_results)

    subset_scores = defaultdict(list)
    for r in detailed_results:
        if "subset" in r:
            subset_scores[r["subset"]].append(r["score"])

    subset_mean_scores = {
        subset_name: sum(scores) / len(scores)
        for subset_name, scores in subset_scores.items()
    }

    # Validate subset keys match current data
    if hint_type is not None:
        examples = eval_data.examples_by_hint_type[hint_type]
    else:
        examples = eval_data.examples

    cached_subset_keys = set(subset_mean_scores.keys())
    current_subset_keys = set(key for key in examples.keys() if key is not None)
    assert (
        cached_subset_keys == current_subset_keys
    ), f"Cached subset keys do not match current subsets: {cached_subset_keys} != {current_subset_keys}"

    return mean_score, subset_mean_scores


def _save_cached_result(cache_file: Path, instructions: str | None, result: EvalResult):
    """Save evaluation result to cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {"instructions": instructions, "evaluations": result.detailed_results}
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


# ============================================================================
# CLI Main Function
# ============================================================================


def main():
    """CLI entry point for generating progression data."""
    import argparse

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
    env = get_environment_from_path(args.experiment_path)
    if env == "mcq":
        print(f"\nLoaded MCQ progression data with {len(data)} hint types:")
        for hint_type, hint_data in data.items():
            print(f"  {hint_type}: {len(hint_data['runs'])} runs")
    else:
        print(f"\nLoaded progression data with {len(data['runs'])} runs")

    print("\nDone!")


if __name__ == "__main__":
    main()
