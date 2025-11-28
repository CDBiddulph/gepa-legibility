"""
Progression data loader for GEPA experiments.

This module provides utilities to load or compute progression data for GEPA experiments,
with automatic caching support.
"""

import json
from pathlib import Path
import dspy
from dotenv import load_dotenv

from core.lm import get_dspy_lm
from core.dspy_utils import get_problem_signature
from core.experiment_utils import (
    get_environment_from_path,
    get_executor_model_name_from_path,
    uses_incompetent_adapter,
)
from tasks.mcq.loader import load_mcq_splits
from tasks.psychosis.loader import load_psychosis_splits
from tasks.wordchain.loader import load_wordchain_splits
from core.progression.evaluation import process_run_progression


# Load environment variables
load_dotenv()


def get_progression_data(experiment_path, quick_mode=False):
    """Load or compute progression data for a single experiment path.

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
    dspy.configure(lm=executor_lm)

    # Load test data
    test_examples = _load_test_data(env)

    # Get metric functions
    metric_fn_proxy, metric_fn_true = _get_metric_functions(env)

    # Get problem signature
    problem_signature = get_problem_signature(env)

    # Collect progression data
    return _collect_progression_data(
        experiment_path,
        env,
        problem_signature,
        metric_fn_proxy,
        metric_fn_true,
        test_examples,
        quick_mode,
    )


# ============================================================================
# Private Helper Functions
# ============================================================================


def _load_test_data(environment):
    """Load test data for the given environment.

    Returns:
        For MCQ: returns empty dict (will be populated per-hint-type)
        For others: returns list of test examples
    """
    if environment == "mcq":
        # For MCQ, we return empty dict - it'll be populated per-hint-type
        return {}
    elif environment.startswith("psychosis"):
        splits = load_psychosis_splits(use_expert_data=False)
        test_examples = splits["test"]
        print(f"Loaded {len(test_examples)} test examples")
        return test_examples
    elif environment == "wordchain":
        splits = load_wordchain_splits(use_expert_data=False)
        test_examples = splits["test"]
        print(f"Loaded {len(test_examples)} test examples")
        return test_examples
    else:
        raise ValueError(f"Unknown environment: {environment}")


def _get_metric_functions(environment):
    """Get proxy and true metric functions for the environment.

    Returns:
        Tuple of (metric_fn_proxy, metric_fn_true)
    """
    if environment == "psychosis_split":
        from tasks.psychosis.scorer import get_metric_fn

        metric_fn_proxy = get_metric_fn(
            judge_model="gpt-4.1-mini", gameable_type="sometimes"
        )
        metric_fn_true = get_metric_fn(
            judge_model="gpt-4.1-mini", gameable_type="never"
        )
    elif environment == "psychosis_single":
        from tasks.psychosis.scorer import get_metric_fn

        metric_fn_proxy = get_metric_fn(
            judge_model="gpt-4.1-mini", gameable_type="never"
        )
        metric_fn_true = get_metric_fn(judge_model="gpt-5", gameable_type="never")
    elif environment == "wordchain":
        from tasks.wordchain.scorer import get_metric_fn

        metric_fn_proxy = get_metric_fn(
            judge_model="gpt-4.1-mini", normalize_response=False
        )
        metric_fn_true = get_metric_fn(
            judge_model="gpt-4.1-mini", normalize_response=True
        )
    elif environment == "mcq":
        from tasks.mcq.scorer import metric_fn as base_mcq_metric_fn

        # Proxy metric uses rewarded_answer as-is
        metric_fn_proxy = base_mcq_metric_fn

        # True metric always uses true_answer as the rewarded answer
        def metric_fn_true(
            example, prediction, trace=None, pred_name=None, pred_trace=None
        ):
            modified_example = dspy.Example(
                question=example.question,
                true_answer=example.true_answer,
                rewarded_answer=example.true_answer,
            ).with_inputs("question")
            return base_mcq_metric_fn(
                modified_example, prediction, trace, pred_name, pred_trace
            )

    else:
        raise ValueError(f"Unknown environment: {environment}")

    return metric_fn_proxy, metric_fn_true


def _collect_progression_data_generic(
    experiment_path,
    test_examples,
    metric_fn_proxy,
    metric_fn_true,
    problem_signature,
    env,
    quick_mode=False,
):
    """Generic progression data collection for a single experiment directory.

    Args:
        experiment_path: Path to experiment directory containing run subdirectories
        test_examples: Test examples for evaluation
        metric_fn_proxy: Proxy metric function
        metric_fn_true: True metric function
        problem_signature: DSPy signature class
        env: Environment name
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
        progression = process_run_progression(
            data,
            existing_progression,
            subdir,
            metric_fn_proxy,
            metric_fn_true,
            test_examples,
            use_incompetent_adapter,
            problem_signature,
            env,
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


def _collect_progression_data_for_hint(
    hint_dir,
    hint_type,
    metric_fn_proxy,
    metric_fn_true,
    problem_signature,
    quick_mode=False,
):
    """Collect progression data for a single MCQ hint type directory.

    Args:
        hint_dir: Path to hint type directory (e.g., .../datamix=mixed/metadata/)
        hint_type: Name of hint type (e.g., 'metadata')
        metric_fn_proxy: Proxy metric function
        metric_fn_true: True metric function
        problem_signature: DSPy signature class
        quick_mode: If True, only evaluate first+last candidates

    Returns:
        Dict with 'runs' key containing list of run data
    """
    print(f"  Collecting progression data for hint type: {hint_type}")

    # Get hint-specific test examples
    test_examples = load_mcq_splits(hint_type, "mixed")["test"]

    return _collect_progression_data_generic(
        hint_dir,
        test_examples,
        metric_fn_proxy,
        metric_fn_true,
        problem_signature,
        env="mcq",
        quick_mode=quick_mode,
    )


def _collect_progression_data(
    experiment_path,
    env,
    problem_signature,
    metric_fn_proxy,
    metric_fn_true,
    test_examples,
    quick_mode=False,
):
    """Collect progression data for a single experiment path.

    Args:
        experiment_path: Path to experiment directory
        env: Environment name
        problem_signature: DSPy signature class
        metric_fn_proxy: Proxy metric function
        metric_fn_true: True metric function
        test_examples: Test examples (or {} for MCQ)
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
            hint_data[hint_type] = _collect_progression_data_for_hint(
                hint_dir,
                hint_type,
                metric_fn_proxy,
                metric_fn_true,
                problem_signature,
                quick_mode=quick_mode,
            )

        return hint_data
    else:
        # Non-MCQ: standard processing
        print(f"Collecting progression data for {experiment_path}")

        return _collect_progression_data_generic(
            experiment_path,
            test_examples,
            metric_fn_proxy,
            metric_fn_true,
            problem_signature,
            env,
            quick_mode=quick_mode,
        )


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
