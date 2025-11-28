"""
Progression evaluation utilities for GEPA experiments.

This module provides functions for evaluating candidates and processing
progression data during GEPA experiment analysis.
"""

import asyncio
import gc
from collections import defaultdict
import json
from itertools import chain

import dspy

from core.instruction_loader import load_instructions
from core.incompetent_adapter import IncompetentAdapter


# ============================================================================
# Public API
# ============================================================================


def process_run_progression(
    data,
    existing_progression,
    subdir,
    metric_fn_proxy,
    metric_fn_true,
    test_examples,
    use_incompetent_adapter,
    problem_signature,
    env,
    quick_mode=False,
):
    """Process progression data for a single run.

    Args:
        data: Loaded detailed_results.json data
        existing_progression: List of already-evaluated progression points
        subdir: Path to run subdirectory
        metric_fn_proxy: Proxy metric function
        metric_fn_true: True metric function
        test_examples: Test examples for evaluation
        use_incompetent_adapter: Whether to configure IncompetentAdapter
        problem_signature: DSPy signature class
        env: Environment name
        quick_mode: If True, only evaluate first and last candidates (for bar plots).
                   If False, evaluate all improving candidates (for progression plots).

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
            metric_fn_proxy,
            metric_fn_true,
            test_examples,
            use_incompetent_adapter,
            problem_signature,
            env,
        )

        prog_point = _create_progression_point(
            candidate, idx, test_scores, subset_test_scores, sanitized_instructions
        )
        progression.append(prog_point)

    # Sort progression by index
    progression.sort(key=lambda x: x["candidate_index"])

    return progression


# ============================================================================
# Private Helper Functions
# ============================================================================


def _get_subsets(environment, examples):
    """Return subset definitions for the environment.

    Returns:
        Dict mapping subset_name -> list of examples.
        Subsets are ordered as they should appear in the output.
    """
    if environment.startswith("psychosis"):
        return {
            "gameable": [ex for ex in examples if ex.is_gameable],
            "ungameable": [ex for ex in examples if not ex.is_gameable],
        }
    elif environment == "mcq":
        return {
            "hinted": [ex for ex in examples if ex.is_hinted],
            "unhinted": [ex for ex in examples if not ex.is_hinted],
        }
    else:
        return {None: examples}


def _get_cached_evaluation_results(cache_file, examples):
    """Load evaluation results from cache file."""
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
    cached_subset_keys = set(subset_mean_scores.keys())
    current_subset_keys = set(key for key in examples.keys() if key is not None)
    assert (
        cached_subset_keys == current_subset_keys
    ), f"Cached subset keys do not match current subsets: {cached_subset_keys} != {current_subset_keys}"
    return mean_score, subset_mean_scores, detailed_results


def _evaluate_program(
    instructions,
    metric_fn,
    examples,
    cache_file,
    use_incompetent_adapter,
    problem_signature,
    env,
):
    """Evaluate using dspy.Evaluate with detailed capture, using cache if available.

    Args:
        instructions: Instructions to use (or None for baseline)
        metric_fn: Metric function to evaluate with
        examples: Mapping from subset name to examples. Examples that don't have a subset should be mapped from None
        cache_file: Path to cache file
        use_incompetent_adapter: Whether to configure IncompetentAdapter
        problem_signature: DSPy signature class
        env: Environment name (e.g., "mcq", "psychosis_single", "psychosis_split")

    Returns:
        Tuple of (aggregate_score, subset_mean_scores, detailed_results)
    """
    # Check cache
    if cache_file.exists():
        return _get_cached_evaluation_results(cache_file, examples)

    # Get current LM from dspy config
    executor_lm = dspy.settings.lm

    # Configure adapter if needed
    if use_incompetent_adapter:
        # Map environment to adapter env
        if env.startswith("psychosis"):
            adapter_env = "psychosis"
        elif env == "mcq":
            adapter_env = "mcq"
        else:
            raise ValueError(f"IncompetentAdapter not supported for environment: {env}")
        adapter = IncompetentAdapter(env=adapter_env)
    else:
        adapter = None
    dspy.configure(lm=executor_lm, adapter=adapter)

    # Create program
    program = dspy.Predict(problem_signature)
    if instructions:
        program.signature = program.signature.with_instructions(instructions)

    # Use dspy.Evaluate to run evaluation
    def evaluate(examples):
        loop = asyncio.get_event_loop()
        try:
            if not loop.is_closed():
                loop.close()
            # Set a fresh event loop for next iteration
            asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError as e:
            print(
                f"Attempted to close event loop to avoid future asyncio errors, but failed: {e}\nThis may be because you are running in a Jupyter notebook. This might be okay; continuing..."
            )
        # Garbage collect to avoid error:
        # `'_UnixSelectorEventLoop' object has no attribute '_ssock'`
        gc.collect()
        return dspy.Evaluate(
            devset=examples,
            metric=metric_fn,
            num_threads=100,
            display_table=False,
            display_progress=True,
        )(program)

    # Condense all examples into a single list
    subset_results = {}
    for subset_name, subset_examples in examples.items():
        subset_results[subset_name] = evaluate(subset_examples)

    # Extract detailed results from evaluation result
    subset_scores = defaultdict(list)
    detailed_results = []
    for subset_name, results in subset_results.items():
        for i, (example, prediction, metric_result) in enumerate(results["results"]):
            # Determine the output field name based on the signature
            if hasattr(prediction, "answer"):
                task_output = prediction.answer
            elif hasattr(prediction, "response"):
                task_output = prediction.response
            else:
                task_output = str(prediction)

            # Handle both Prediction objects and raw float scores
            # (raw floats occur when dspy.Evaluate encounters a failure)
            if isinstance(metric_result, float):
                score = metric_result
                feedback = "Evaluation failed"
            else:
                score = metric_result.score
                feedback = metric_result.feedback

            detailed_result = {
                "example_index": i,
                "example_inputs": dict(example.inputs()),
                "task_lm_output": task_output,
                "score": score,
                "feedback": feedback,
            }
            if subset_name:
                detailed_result["subset"] = subset_name
            detailed_results.append(detailed_result)
            subset_scores[subset_name].append(score)

    # Save cache with instructions at top level
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {"instructions": instructions, "evaluations": detailed_results}
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    all_scores = list(chain(*subset_scores.values()))
    mean_score = sum(all_scores) / len(all_scores)
    subset_mean_scores = {
        subset_name: sum(scores) / len(scores)
        for subset_name, scores in subset_scores.items()
    }

    return mean_score, subset_mean_scores, detailed_results


def _create_progression_point(
    candidate, idx, test_scores, subset_test_scores, sanitized_instructions
):
    """Create a progression point dict with proper field ordering.

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


def _evaluate_candidate(
    candidate,
    idx,
    subdir,
    metric_fn_proxy,
    metric_fn_true,
    test_examples,
    use_incompetent_adapter,
    problem_signature,
    env,
):
    """Evaluate a single candidate on all test conditions with detailed capture.

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

    # Get subsets for this environment
    subsets = _get_subsets(env, test_examples)

    # Evaluate all version × metric combinations
    metrics = [("proxy", metric_fn_proxy), ("true", metric_fn_true)]

    for version_name, instructions in versions:
        for metric_type, metric_fn in metrics:
            cache_file = eval_dir / f"cand_{idx}_{metric_type}_{version_name}.json"
            mean_score, subset_mean_scores, _ = _evaluate_program(
                instructions,
                metric_fn,
                subsets,
                cache_file,
                use_incompetent_adapter,
                problem_signature,
                env,
            )
            score_key = f"{metric_type}_{version_name}"
            test_scores[score_key] = mean_score
            # Collect subset scores
            for subset_name, subset_score in subset_mean_scores.items():
                if subset_name is not None:
                    subset_test_scores[subset_name][score_key] = subset_score

    return test_scores, subset_test_scores, sanitized_instructions
