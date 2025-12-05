"""Core GEPA runner logic shared across tasks."""

import json
import os
import traceback
from pathlib import Path
from typing import Callable

import dspy

from core.config import save_config
from core.reflection import get_reflection_prompt_template
from core.incompetent_adapter import IncompetentAdapter
from core.instruction_proposer import CustomPromptInstructionProposer
from core.lm import get_dspy_lm
from core.logging_utils import gepa_logging, serialize_detailed_results
from core.evaluation import get_progression_data
from scoring.length_penalty import length_penalty_metric_fn


# Monkey-patch GEPA to always capture traces during evaluation.
#
# DspyAdapter.evaluate only captures traces during reflection, not evaluation.
# This patch forces trace capture so metrics (like length_penalty) can access instruction text.
def _patch_gepa_trace_capture():
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    original_evaluate = DspyAdapter.evaluate

    def evaluate_with_traces(self, batch, candidate, capture_traces=False):
        return original_evaluate(self, batch, candidate, capture_traces=True)

    DspyAdapter.evaluate = evaluate_with_traces


_patch_gepa_trace_capture()


def _load_baseline_instructions(baseline_instructions_path: str | None) -> str | None:
    """Load baseline instructions from a file or directory.

    Args:
        baseline_instructions_path: Path to either:
            - A .txt file containing instructions
            - A directory containing best_instructions.txt
            Or None to skip loading.

    Returns:
        The instructions string, or None if no path specified.

    Raises:
        FileNotFoundError: If path specified but file not found.
    """
    if baseline_instructions_path is None:
        return None

    path = Path(baseline_instructions_path)

    # If it's a .txt file, read it directly
    if path.suffix == ".txt" and not path.is_dir():
        if not path.exists():
            raise FileNotFoundError(f"Instructions file not found at {path}")
        with open(path, "r") as f:
            return f.read()

    # Otherwise, look for best_instructions.txt in the directory
    instructions_path = path / "best_instructions.txt"
    if not instructions_path.exists():
        raise FileNotFoundError(
            f"best_instructions.txt not found at {instructions_path}"
        )

    with open(instructions_path, "r") as f:
        return f.read()


def run_gepa(
    config,
    dataset: dict,
    metric_fn: Callable,
    signature_class: type,
    log_dir: str,
    num_threads: int = 32,
    reflection_minibatch_size: int = 10,
) -> None:
    """Run GEPA optimization for a single trial.

    Args:
        config: Task-specific configuration dataclass.
        dataset: Dict with 'train', 'valid', 'test' keys containing example lists
        metric_fn: Base metric function for evaluation (without length penalty)
        signature_class: DSPy Signature class for the task
        log_dir: Directory to save results
        num_threads: Number of threads for parallel evaluation
        reflection_minibatch_size: Minibatch size for reflection
    """
    if os.path.exists(os.path.join(log_dir, "detailed_results.json")):
        print(
            f"Skipping GEPA for {log_dir} because detailed_results.json already exists"
        )
        return

    print("Saving logs to:", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    save_config(log_dir, config)

    try:
        prompter_lm = get_dspy_lm(config.prompter_name, cache=config.cache)
        executor_lm = get_dspy_lm(
            config.executor_name,
            cache=config.cache,
            reasoning_effort=config.executor_reasoning_effort,
        )

        # Configure DSPy with IncompetentAdapter if incompetent is True
        adapter = None
        if config.incompetent:
            env_name = getattr(config, "env_name", None)
            if env_name:
                adapter = IncompetentAdapter(env=env_name)
            else:
                adapter = IncompetentAdapter()
            print("Using IncompetentAdapter to make LM depend on written strategies")
        dspy.configure(lm=executor_lm, adapter=adapter)

        valset = dataset["valid"]
        if config.validation_set_size > len(valset):
            raise ValueError(
                f"Validation set size {config.validation_set_size} is greater than "
                f"the number of validation examples {len(valset)}"
            )

        # Wrap metric with length penalty
        # If length penalty is 0.0, this has no effect
        metric_fn = length_penalty_metric_fn(metric_fn, config.length_penalty)

        # Create custom instruction proposer with reflection template
        use_teacher = getattr(config, "use_teacher", False)
        reflection_template = get_reflection_prompt_template(
            config.suggest_hack, use_teacher, config.length_penalty
        )
        custom_proposer = CustomPromptInstructionProposer(
            reflection_lm=prompter_lm,
            prompt_template=reflection_template,
        )

        optimizer = dspy.GEPA(
            metric=metric_fn,
            max_metric_calls=config.max_metric_calls,
            num_threads=num_threads,
            track_stats=True,
            reflection_minibatch_size=reflection_minibatch_size,
            instruction_proposer=custom_proposer,
            log_dir=log_dir,
            use_merge=True,
            max_merge_invocations=5,
            seed=config.seed,
        )

        # Load baseline instructions if directory specified
        baseline_instructions = _load_baseline_instructions(
            config.baseline_instructions_path
        )
        if baseline_instructions is not None:
            print(f"Using baseline instructions from: {config.baseline_instructions_path}")
            signature_class = signature_class.with_instructions(baseline_instructions)
        baseline_program = dspy.Predict(signature_class)

        with gepa_logging(os.path.join(log_dir, "gepa.log")):
            optimized_program = optimizer.compile(
                baseline_program,
                trainset=dataset["train"],
                valset=valset[: config.validation_set_size],
            )

        with open(os.path.join(log_dir, "best_instructions.txt"), "w") as f:
            f.write(optimized_program.signature.instructions)

        print("Completed optimization. Evaluating...")

        _eval_and_save_detailed_results(
            log_dir=log_dir,
            detailed_results=optimized_program.detailed_results,
            log_dir_index=config.log_dir_index,
            prompter_history=prompter_lm.history,
        )
    except Exception as e:
        error_message = f"Error running GEPA: {e}\n\n{traceback.format_exc()}"
        print(error_message)
        with open(os.path.join(log_dir, "detailed_results.err"), "w") as f:
            f.write(error_message)
        raise


def _eval_and_save_detailed_results(
    log_dir: str,
    detailed_results,
    log_dir_index: int,
    prompter_history: list,
) -> None:
    """Evaluate on test set and save detailed results.

    Args:
        log_dir: Directory containing this trial's results
        detailed_results: GEPA detailed results object
        log_dir_index: Index of this trial
        prompter_history: History from the prompter LM
    """
    # Save minimal detailed_results.json for get_progression_data to use
    minimal_serialized_results = serialize_detailed_results(
        detailed_results,
        "Waiting for results...",
        "Waiting for results...",
        prompter_history,
    )
    detailed_results_path = os.path.join(log_dir, "detailed_results.json")
    with open(detailed_results_path, "w") as f:
        json.dump(minimal_serialized_results, f, indent=2)
        print(f"Saved minimal detailed results to {detailed_results_path}")

    # Use progression_loader to get test scores with caching
    experiment_path = Path(log_dir).parent
    progression_data = get_progression_data(str(experiment_path), quick_mode=True)

    # Find run by run_index since list position may differ if some runs failed
    run_data = next(
        (r for r in progression_data["runs"] if r["run_index"] == log_dir_index),
        None,
    )
    if run_data is None:
        raise ValueError(
            f"Run {log_dir_index} not found in progression data. "
            f"Available runs: {[r['run_index'] for r in progression_data['runs']]}"
        )

    # Find baseline (candidate 0) and best candidate
    baseline_point = next(
        p for p in run_data["progression"] if p["candidate_index"] == 0
    )
    best_point = max(run_data["progression"], key=lambda p: p["validation_score"])

    # Extract test scores (using proxy_original to match training metric)
    baseline_test_score = baseline_point["test_scores"]["proxy_original"]
    best_test_score = best_point["test_scores"]["proxy_original"]

    # Format subset scores in requested structure
    subset_test_scores = {}
    if "subset_test_scores" in best_point:
        for subset_name in best_point["subset_test_scores"].keys():
            subset_test_scores[subset_name] = {
                "best": best_point["subset_test_scores"][subset_name]["proxy_original"],
                "baseline": baseline_point["subset_test_scores"][subset_name][
                    "proxy_original"
                ],
            }

    # Construct path to full progression data file
    progression_data_file = str(
        experiment_path / str(log_dir_index) / "progression_data.json"
    )

    print(f"Best test score: {best_test_score:.3f}")
    print(f"Baseline test score: {baseline_test_score:.3f}")

    serialized_results = serialize_detailed_results(
        detailed_results,
        best_test_score,
        baseline_test_score,
        prompter_history,
        subset_test_scores=subset_test_scores,
        progression_data_file=progression_data_file,
    )

    with open(detailed_results_path, "w") as f:
        json.dump(serialized_results, f, indent=2)
        print(f"Saved detailed results to {detailed_results_path}")
