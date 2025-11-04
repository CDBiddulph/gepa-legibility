"""
Progression data loader for GEPA experiments.

This module provides utilities to load or compute progression data for GEPA experiments,
with automatic caching support.
"""

import json
from pathlib import Path
import dspy
from dotenv import load_dotenv

from instruction_loader import load_instructions
from incompetent_adapter import IncompetentAdapter
from lm import get_dspy_lm


# Load environment variables
load_dotenv()

# Constants
PLOT_DATA_BASE = Path("plot-data")


# ============================================================================
# Public API Functions
# ============================================================================


def get_environment_from_path(path):
    """Extract environment type from experiment path.

    Args:
        path: Path to experiment directory

    Returns:
        Environment name: 'psychosis_single', 'psychosis_split', 'mcq', 'alliterate', or 'wordchain'
    """
    path = str(path)
    if "psychosis" in path:
        if "gameable=never" in path:
            return "psychosis_single"
        elif "gameable=sometimes" in path or "gameable=" not in path:
            return "psychosis_split"
        else:
            raise ValueError(f"Unknown psychosis setting for path: {path}")
    elif "mcq" in path:
        return "mcq"
    elif "alliterate" in path:
        return "alliterate"
    elif "wordchain" in path:
        return "wordchain"
    else:
        raise ValueError(f"Unknown setting for path: {path}")


def get_executor_model_name_from_path(path):
    """Extract executor model name from experiment path.

    Args:
        path: Path to experiment directory

    Returns:
        Model name in format 'openai/model-name'
    """
    executor_substr = str(path).split("e=")[-1]
    if executor_substr.startswith("gpt-4.1-mini"):
        return "openai/gpt-4.1-mini"
    elif executor_substr.startswith("o4-mini"):
        return "openai/o4-mini"
    elif executor_substr.startswith("gpt-5-mini"):
        return "openai/gpt-5-mini"
    else:
        raise ValueError(f"Unknown executor: {executor_substr}")


def uses_incompetent_adapter(experiment_path):
    """Check if an experiment path indicates IncompetentAdapter should be used.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Boolean indicating whether IncompetentAdapter should be used
    """
    return "forget" in str(experiment_path).lower()


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
    problem_signature = _get_problem_signature(env)

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


def _get_test_example_from_dict(data, environment):
    """Convert dict data to dspy.Example."""
    if environment.startswith("psychosis"):
        return dspy.Example(
            memory=data["memory"], query=data["query"], is_gameable=data["is_gameable"]
        ).with_inputs("memory", "query")
    elif environment == "alliterate":
        return dspy.Example(
            query=data["query"], sentence=data["sentence"], letter=data["letter"]
        ).with_inputs("query")
    elif environment == "wordchain":
        return dspy.Example(
            query=data["query"],
            start_word=data["start_word"],
            end_word=data["end_word"],
        ).with_inputs("query")
    elif environment == "mcq":
        return dspy.Example(
            question=data["question"],
            true_answer=data["true_answer"],
            rewarded_answer=data["rewarded_answer"],
        ).with_inputs("question")
    else:
        raise ValueError(f"Unknown environment: {environment}")


def _load_jsonl_test_data(file_path, environment):
    """Load test data from JSONL file."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(_get_test_example_from_dict(data, environment))
    return examples


def _load_mcq_mixed_test_data(hint_type):
    """Load mixed MCQ test data for a specific hint type.

    Reconstructs mixed test data: first half from correct/none, second half from incorrect/hint_type.

    Args:
        hint_type: The hint type (e.g., 'metadata', 'sycophancy')

    Returns:
        List of test examples
    """
    data_dir = Path("data/mcq")

    # Load both correct (none) and incorrect (hint_type) data
    correct_data = []
    incorrect_data = []

    with open(data_dir / "none" / "test.jsonl", "r") as f:
        for line in f:
            correct_data.append(json.loads(line.strip()))

    with open(data_dir / hint_type / "test.jsonl", "r") as f:
        for line in f:
            incorrect_data.append(json.loads(line.strip()))

    # Create mixed dataset: first half correct/unhinted, second half incorrect/hinted
    midpoint = len(correct_data) // 2

    mixed_examples = []

    # First half: correct_hinted_input from none, rewarded = correct
    for i in range(midpoint):
        data = correct_data[i]
        mixed_examples.append(
            {
                "question": data["correct_hinted_input"],
                "true_answer": data["correct_answer"],
                "rewarded_answer": data["correct_answer"],
            }
        )

    # Second half: incorrect_hinted_input from hint_type, rewarded = incorrect
    for i in range(midpoint, len(incorrect_data)):
        data = incorrect_data[i]
        mixed_examples.append(
            {
                "question": data["incorrect_hinted_input"],
                "true_answer": data["correct_answer"],
                "rewarded_answer": data["incorrect_answer"],
            }
        )

    # Create examples
    examples = [
        dspy.Example(**ex).with_inputs("question") for ex in mixed_examples
    ]

    return examples


def _discover_hint_types_from_path(exp_path):
    """Discover hint type subdirectories from an experiment path."""
    hint_types = []
    if exp_path.exists() and exp_path.is_dir():
        for subdir in exp_path.iterdir():
            if subdir.is_dir():
                hint_types.append(subdir.name)
    return sorted(hint_types)


def _load_test_data(environment):
    """Load test data for the given environment.

    Returns:
        For MCQ: returns empty dict (will be populated per-hint-type)
        For others: returns list of test examples
    """
    if environment == "mcq":
        # For MCQ, we return empty dict - it'll be populated per-hint-type
        return {}
    else:
        environment_for_data_path = (
            "psychosis" if environment.startswith("psychosis") else environment
        )
        data_path = Path(f"data/{environment_for_data_path}/test.jsonl")
        test_examples = _load_jsonl_test_data(data_path, environment)
        print(f"Loaded {len(test_examples)} test examples")
        return test_examples


def _get_metric_functions(environment):
    """Get proxy and true metric functions for the environment.

    Returns:
        Tuple of (metric_fn_proxy, metric_fn_true)
    """
    if environment == "psychosis_split":
        from scoring.psychosis_scale import get_metric_fn

        metric_fn_proxy = get_metric_fn(
            judge_model="gpt-4.1-mini", gameable_type="sometimes"
        )
        metric_fn_true = get_metric_fn(
            judge_model="gpt-4.1-mini", gameable_type="never"
        )
    elif environment == "psychosis_single":
        from scoring.psychosis_scale import get_metric_fn

        metric_fn_proxy = get_metric_fn(
            judge_model="gpt-4.1-mini", gameable_type="never"
        )
        metric_fn_true = get_metric_fn(judge_model="gpt-5", gameable_type="never")
    elif environment == "alliterate":
        from scoring.alliterate import get_metric_fn

        metric_fn_proxy = get_metric_fn(judge_model="gpt-4.1-mini", only_rewrite=False)
        metric_fn_true = get_metric_fn(judge_model="gpt-4.1-mini", only_rewrite=True)
    elif environment == "wordchain":
        from scoring.wordchain import get_metric_fn

        metric_fn_proxy = get_metric_fn(judge_model="gpt-4.1-mini", only_answer=False)
        metric_fn_true = get_metric_fn(judge_model="gpt-4.1-mini", only_answer=True)
    elif environment == "mcq":
        from scoring.mcq import metric_fn as base_mcq_metric_fn

        def metric_fn_proxy(
            example, prediction, trace=None, pred_name=None, pred_trace=None
        ):
            return base_mcq_metric_fn(example, prediction, trace, pred_name, pred_trace)

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


def _get_problem_signature(environment):
    """Get the DSPy signature class for the environment."""
    if environment.startswith("psychosis"):

        class ProblemSignature(dspy.Signature):
            memory = dspy.InputField(
                description="The memory of the AI assistant about the user"
            )
            query = dspy.InputField(description="The user's query")
            response = dspy.OutputField(description="The response of the AI assistant.")

    elif environment == "alliterate":

        class ProblemSignature(dspy.Signature):
            query = dspy.InputField(description="The query")
            response = dspy.OutputField(description="The response to the query")

    elif environment == "wordchain":

        class ProblemSignature(dspy.Signature):
            query = dspy.InputField(description="The query")
            response = dspy.OutputField(description="The response to the query")

    elif environment == "mcq":

        class ProblemSignature(dspy.Signature):
            question = dspy.InputField(description="The multiple choice question")
            answer = dspy.OutputField(
                description="The answer, in the format 'Answer: <answer>'. You can also include reasoning in this field, but it is not required."
            )

    else:
        raise ValueError(f"Unknown environment: {environment}")

    return ProblemSignature


def _evaluate_program(
    instructions,
    metric_fn,
    examples,
    cache_file,
    use_incompetent_adapter,
    problem_signature,
):
    """Evaluate using dspy.Evaluate with detailed capture, using cache if available.

    Args:
        instructions: Instructions to use (or None for baseline)
        metric_fn: Metric function to evaluate with
        examples: Test examples
        cache_file: Path to cache file
        use_incompetent_adapter: Whether to configure IncompetentAdapter
        problem_signature: DSPy signature class

    Returns:
        Tuple of (aggregate_score, detailed_results)
    """
    # Check cache
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        detailed_results = data["evaluations"]
        score = sum(r["score"] for r in detailed_results) / len(detailed_results)
        return score, detailed_results

    # Get current LM from dspy config
    executor_lm = dspy.settings.lm

    # Configure adapter if needed
    adapter = IncompetentAdapter() if use_incompetent_adapter else None
    dspy.configure(lm=executor_lm, adapter=adapter)

    # Create program
    program = dspy.Predict(problem_signature)
    if instructions:
        program.signature = program.signature.with_instructions(instructions)

    # Use dspy.Evaluate to run evaluation
    evaluate = dspy.Evaluate(
        devset=examples,
        metric=metric_fn,
        num_threads=80,
        display_table=False,
        display_progress=True,
    )
    result = evaluate(program)

    # Extract detailed results from evaluation result
    detailed_results = []
    for i, (example, prediction, metric_result) in enumerate(result["results"]):
        # Determine the output field name based on the signature
        if hasattr(prediction, "answer"):
            task_output = prediction.answer
        elif hasattr(prediction, "response"):
            task_output = prediction.response
        else:
            task_output = str(prediction)

        detailed_results.append(
            {
                "example_index": i,
                "example_inputs": dict(example.inputs()),
                "task_lm_output": task_output,
                "score": metric_result.score,
                "feedback": metric_result.feedback,
            }
        )

    # Save cache with instructions at top level
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {"instructions": instructions, "evaluations": detailed_results}
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    # Return aggregate score from dspy.Evaluate result
    aggregate_score = result.score / 100.0  # dspy.Evaluate returns percentage
    return aggregate_score, detailed_results


def _evaluate_candidate(
    candidate,
    idx,
    subdir,
    metric_fn_proxy,
    metric_fn_true,
    test_examples,
    plot_data_path,
    use_incompetent_adapter,
    problem_signature,
):
    """Evaluate a single candidate on all test conditions with detailed capture.

    Returns:
        Tuple of (test_scores, sanitized_instructions)
    """
    instructions = candidate["instructions"]
    test_scores = {}

    # Build evaluation directory path
    eval_dir = plot_data_path / "evaluations" / f"run_{subdir.name}"

    # Baseline case (candidate 0 has no instructions)
    if idx == 0:
        instructions_for_eval = None

        # Proxy original
        cache_file = eval_dir / f"cand_{idx}_proxy_original.json"
        score, _ = _evaluate_program(
            instructions_for_eval,
            metric_fn_proxy,
            test_examples,
            cache_file,
            use_incompetent_adapter,
            problem_signature,
        )
        test_scores["proxy_original"] = score

        # True original
        cache_file = eval_dir / f"cand_{idx}_true_original.json"
        score, _ = _evaluate_program(
            instructions_for_eval,
            metric_fn_true,
            test_examples,
            cache_file,
            use_incompetent_adapter,
            problem_signature,
        )
        test_scores["true_original"] = score

        # Same for baseline
        test_scores["proxy_sanitized"] = test_scores["proxy_original"]
        test_scores["true_sanitized"] = test_scores["true_original"]
        sanitized_instructions = None
    else:
        # Original instructions
        cache_file = eval_dir / f"cand_{idx}_proxy_original.json"
        score, _ = _evaluate_program(
            instructions,
            metric_fn_proxy,
            test_examples,
            cache_file,
            use_incompetent_adapter,
            problem_signature,
        )
        test_scores["proxy_original"] = score

        cache_file = eval_dir / f"cand_{idx}_true_original.json"
        score, _ = _evaluate_program(
            instructions,
            metric_fn_true,
            test_examples,
            cache_file,
            use_incompetent_adapter,
            problem_signature,
        )
        test_scores["true_original"] = score

        # Sanitized instructions
        sanitized_instructions = load_instructions(subdir, idx, sanitized=True)

        cache_file = eval_dir / f"cand_{idx}_proxy_sanitized.json"
        score, _ = _evaluate_program(
            sanitized_instructions,
            metric_fn_proxy,
            test_examples,
            cache_file,
            use_incompetent_adapter,
            problem_signature,
        )
        test_scores["proxy_sanitized"] = score

        cache_file = eval_dir / f"cand_{idx}_true_sanitized.json"
        score, _ = _evaluate_program(
            sanitized_instructions,
            metric_fn_true,
            test_examples,
            cache_file,
            use_incompetent_adapter,
            problem_signature,
        )
        test_scores["true_sanitized"] = score

    return test_scores, sanitized_instructions


def _process_run_progression(
    data,
    existing_progression,
    subdir,
    metric_fn_proxy,
    metric_fn_true,
    test_examples,
    plot_data_path,
    use_incompetent_adapter,
    problem_signature,
    quick_mode=False,
):
    """Process progression data for a single run.

    Args:
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
        test_scores, sanitized_instructions = _evaluate_candidate(
            candidate,
            idx,
            subdir,
            metric_fn_proxy,
            metric_fn_true,
            test_examples,
            plot_data_path,
            use_incompetent_adapter,
            problem_signature,
        )

        progression.append(
            {
                "candidate_index": idx,
                "validation_score": val_score,
                "discovery_eval_counts": candidate["discovery_eval_counts"],
                "reflection_call_count": candidate["reflection_call_count"],
                "test_scores": test_scores,
                "original_instructions": candidate["instructions"],
                "sanitized_instructions": sanitized_instructions,
            }
        )

    # Sort progression by index
    progression.sort(key=lambda x: x["candidate_index"])

    return progression


def _collect_progression_data_generic(
    experiment_path,
    plot_data_path,
    test_examples,
    metric_fn_proxy,
    metric_fn_true,
    problem_signature,
    quick_mode=False,
):
    """Generic progression data collection for a single experiment directory.

    Args:
        experiment_path: Path to experiment directory containing run subdirectories
        plot_data_path: Path where to save/load cached progression data
        test_examples: Test examples for evaluation
        metric_fn_proxy: Proxy metric function
        metric_fn_true: True metric function
        problem_signature: DSPy signature class
        quick_mode: If True, only evaluate first+last (for bar plots)

    Returns:
        Dict with 'runs' key containing list of run data
    """
    # Check if this experiment uses IncompetentAdapter
    use_incompetent_adapter = uses_incompetent_adapter(experiment_path)
    if use_incompetent_adapter:
        print(
            f"  Detected 'forget' in path - will use IncompetentAdapter during evaluation"
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
        run_file = plot_data_path / f"progression_data_{run_index}.json"

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
            metric_fn_proxy,
            metric_fn_true,
            test_examples,
            plot_data_path,
            use_incompetent_adapter,
            problem_signature,
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

        # Save individual run file
        plot_data_path.mkdir(parents=True, exist_ok=True)
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

    # Create plot data path at hint type level
    relative_path = Path(*hint_dir.parts[1:])  # Remove 'logs' prefix
    plot_data_path = PLOT_DATA_BASE / relative_path

    # Get hint-specific test examples
    test_examples = _load_mcq_mixed_test_data(hint_type)

    return _collect_progression_data_generic(
        hint_dir,
        plot_data_path,
        test_examples,
        metric_fn_proxy,
        metric_fn_true,
        problem_signature,
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

        relative_path = Path(*experiment_path.parts[1:])  # Remove 'logs' prefix
        plot_data_path = PLOT_DATA_BASE / relative_path

        return _collect_progression_data_generic(
            experiment_path,
            plot_data_path,
            test_examples,
            metric_fn_proxy,
            metric_fn_true,
            problem_signature,
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
