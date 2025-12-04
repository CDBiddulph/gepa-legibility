"""Core GEPA runner logic shared across tasks."""

import json
import os
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


# Monkey-patch GEPA to always capture traces AND pass them to metrics.
#
# There are two issues in DSPy's GEPA implementation:
# 1. DspyAdapter.evaluate only captures traces during reflection, not evaluation
# 2. bootstrap_trace_data captures traces but discards them in wrapped_metric
#
# This patch fixes both so metrics (like length_penalty) can access instruction text.
# See: https://github.com/stanfordnlp/dspy/commit/e78c1058d58dae5c2629d8f4cba26ad6f6b42834
def _patch_gepa_trace_handling():
    from types import MethodType
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module
    from dspy.evaluate.evaluate import Evaluate
    from dspy.utils.exceptions import AdapterParseError
    import dspy
    import logging

    logger = logging.getLogger(__name__)

    # Patch 1: Always capture traces in DspyAdapter.evaluate
    original_evaluate = DspyAdapter.evaluate

    def evaluate_with_traces(self, batch, candidate, capture_traces=False):
        return original_evaluate(self, batch, candidate, capture_traces=True)

    DspyAdapter.evaluate = evaluate_with_traces

    # Patch 2: Replace bootstrap_trace_data with a fixed version that passes trace to metric
    # The original has a bug where wrapped_metric discards the trace
    def fixed_bootstrap_trace_data(
        program, dataset, metric=None, num_threads=None, raise_on_error=True,
        capture_failed_parses=False, failure_score=0, format_failure_score=-1,
        log_format_failures=False, callback_metadata=None,
    ):
        FailedPrediction = bootstrap_trace_module.FailedPrediction

        evaluator = Evaluate(
            devset=dataset,
            num_threads=num_threads,
            display_progress=True,
            provide_traceback=False,
            max_errors=len(dataset) * 10,
            failure_score=failure_score,
        )

        # FIXED: Pass the trace to the metric instead of discarding it
        def wrapped_metric(example, prediction, trace=None):
            prediction, captured_trace = prediction  # Unpack tuple
            if isinstance(prediction, FailedPrediction):
                return prediction.format_reward or format_failure_score
            # FIXED: Pass captured_trace instead of None
            return metric(example, prediction, captured_trace) if metric else True

        original_forward = object.__getattribute__(program, "forward")

        def patched_forward(program_to_use, **kwargs):
            with dspy.context(trace=[]):
                try:
                    return original_forward(**kwargs), dspy.settings.trace.copy()
                except AdapterParseError as e:
                    completion_str = e.lm_response
                    parsed_result = e.parsed_result
                    failed_signature = e.signature
                    failed_inputs = kwargs

                    present = list(parsed_result.keys()) if parsed_result else None
                    expected = list(failed_signature.output_fields.keys())

                    found_pred = None
                    for pred in program_to_use.predictors():
                        if pred.signature == failed_signature:
                            found_pred = pred
                            break
                    if found_pred is None:
                        raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                    trace = dspy.settings.trace.copy()
                    if present:
                        failed_pred = FailedPrediction(
                            completion_text=completion_str,
                            format_reward=format_failure_score
                            + (failure_score - format_failure_score) * (len(present) / len(expected)),
                        )
                    else:
                        failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                    trace.append((found_pred, failed_inputs, failed_pred))

                    if log_format_failures:
                        logger.warning(
                            "Failed to parse output for example. This is likely due to the LLM response not following "
                            "the adapter's formatting."
                        )

                    return failed_pred, trace

        program.forward = MethodType(patched_forward, program)

        try:
            results = evaluator(
                program,
                metric=wrapped_metric,
                callback_metadata=callback_metadata,
            ).results
        finally:
            program.forward = original_forward

        data = []
        for example_ind, (example, prediction, score) in enumerate(results):
            try:
                prediction, trace = prediction
            except (ValueError, TypeError) as ve:
                logger.warning(
                    "Failed to unpack prediction and trace. This is likely due to the LLM response not following "
                    "dspy formatting."
                )
                if raise_on_error:
                    raise ve
                else:
                    continue
            data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
            if metric:
                data_dict["score"] = score
            data.append(data_dict)

        return data

    bootstrap_trace_module.bootstrap_trace_data = fixed_bootstrap_trace_data
    print("[PATCH] GEPA trace handling patched successfully")


_patch_gepa_trace_handling()


def run_gepa(
    config,
    dataset: dict,
    metric_fn: Callable,
    signature_class: type,
    log_dir: str,
    get_progression_path: Callable[[str], Path],
    get_progression_key: Callable[[dict, int], dict] = None,
    reflection_template: str = None,
    num_threads: int = 32,
    reflection_minibatch_size: int = 10,
    executor_reasoning_effort: str = "low",
) -> None:
    """Run GEPA optimization for a single trial.

    Args:
        config: Task-specific configuration dataclass
        dataset: Dict with 'train', 'valid', 'test' keys containing example lists
        metric_fn: Metric function for evaluation
        signature_class: DSPy Signature class for the task
        log_dir: Directory to save results
        get_progression_path: Function that takes log_dir and returns the experiment
            path for progression_loader
        get_progression_key: Optional function that takes (progression_data, log_dir_index)
            and returns the run data dict. If None, uses progression_data["runs"][index].
        reflection_template: Custom reflection template. If None, uses default based on
            config.suggest_hack.
        num_threads: Number of threads for parallel evaluation
        reflection_minibatch_size: Minibatch size for reflection
        executor_reasoning_effort: Reasoning effort for executor LM
    """
    if os.path.exists(os.path.join(log_dir, "detailed_results.json")):
        print(
            f"Skipping GEPA for {log_dir} because detailed_results.json already exists"
        )
        return

    print("Saving logs to:", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    save_config(log_dir, config)

    prompter_lm = get_dspy_lm(config.prompter_name, cache=config.cache)
    executor_lm = get_dspy_lm(
        config.executor_name,
        cache=config.cache,
        reasoning_effort=executor_reasoning_effort,
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

    # Create custom instruction proposer
    template = reflection_template or get_reflection_prompt_template(
        config.suggest_hack
    )
    custom_proposer = CustomPromptInstructionProposer(
        reflection_lm=prompter_lm,
        prompt_template=template,
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
        get_progression_path=get_progression_path,
        get_progression_key=get_progression_key,
    )


def _eval_and_save_detailed_results(
    log_dir: str,
    detailed_results,
    log_dir_index: int,
    prompter_history: list,
    get_progression_path: Callable[[str], Path],
    get_progression_key: Callable[[dict, int], dict] = None,
) -> None:
    """Evaluate on test set and save detailed results.

    Args:
        log_dir: Directory containing this trial's results
        detailed_results: GEPA detailed results object
        log_dir_index: Index of this trial
        prompter_history: History from the prompter LM
        get_progression_path: Function to get experiment path from log_dir
        get_progression_key: Function to extract run data from progression data
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
    experiment_path = get_progression_path(log_dir)
    progression_data = get_progression_data(str(experiment_path), quick_mode=True)

    # Extract this specific run's data
    if get_progression_key:
        run_data = get_progression_key(progression_data, log_dir_index)
    else:
        run_data = progression_data["runs"][log_dir_index]

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
