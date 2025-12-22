"""
Core evaluation functions.

This module provides the main evaluate() function for running evaluations
using dspy.Evaluate.
"""

import asyncio
import gc
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Literal

import dspy

from core.evaluation.loaders import EvalData, ExamplesBySubset
from core.lm import extract_reasoning_from_history_entry, find_and_pop_history_entry


@dataclass
class EvalResult:
    """Result from evaluate()."""

    score: float
    subset_scores: dict[str, float]  # e.g., {"gameable": 0.8} or {} if no subsets
    detailed_results: list[dict]


def evaluate(
    data: EvalData,
    lm,
    reward: Literal["proxy", "true"],
    *,
    instructions: str | None = None,
    hint_type: str | None = None,
    adapter: Literal["chat", "incompetent"] = "chat",
    num_threads: int = 100,
) -> EvalResult:
    """
    Evaluate a program on examples.

    Args:
        data: EvalData containing examples, reward functions, signature, adapters
        lm: DSPy language model to use
        reward: Which reward function to use ("proxy" or "true")
        instructions: Optional instructions to add to the program
        hint_type: For MCQ only - which hint type to evaluate on
        adapter: Which adapter to use ("chat" or "incompetent")
        num_threads: Number of threads for dspy.Evaluate

    Returns:
        EvalResult with score, subset scores, and detailed results.
        If the LM produces reasoning traces, they are included in detailed_results.
    """
    # Get examples based on whether hint_type is provided
    if hint_type is not None:
        examples = data.examples_by_hint_type[hint_type]
    else:
        examples = data.examples

    # Get reward function and adapter
    reward_fn = data.reward_fns[reward]
    adapter_instance = data.adapters[adapter]

    # Configure DSPy
    dspy.configure(lm=lm, adapter=adapter_instance)

    # Create program
    program = dspy.Predict(data.signature)
    if instructions:
        program.signature = program.signature.with_instructions(instructions)

    # Run evaluation on each subset
    return _run_evaluation(program, examples, reward_fn, num_threads, lm)


def _reset_event_loop():
    """
    Reset asyncio event loop to avoid errors with dspy.Evaluate.

    This is needed because dspy.Evaluate can leave the event loop in a bad state,
    causing errors on subsequent evaluations.
    """
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError as e:
        print(
            f"Attempted to close event loop to avoid future asyncio errors, but failed: {e}\n"
            "This may be because you are running in a Jupyter notebook. This might be okay; continuing..."
        )
    # Garbage collect to avoid error:
    # `'_UnixSelectorEventLoop' object has no attribute '_ssock'`
    gc.collect()


def _run_evaluation(
    program,
    examples: ExamplesBySubset,
    reward_fn,
    num_threads: int,
    lm,
) -> EvalResult:
    """
    Run dspy.Evaluate on examples organized by subset.

    Args:
        program: DSPy program to evaluate
        examples: Dict mapping subset name (or None) to list of examples
        reward_fn: Reward function to use
        num_threads: Number of threads for parallel evaluation
        lm: Language model (used to extract reasoning from history)

    Returns:
        EvalResult with aggregated scores and detailed results
    """
    _reset_event_loop()

    # Run evaluation for each subset
    subset_results = {}
    for subset_name, subset_examples in examples.items():
        evaluator = dspy.Evaluate(
            devset=subset_examples,
            metric=reward_fn,
            num_threads=num_threads,
            display_table=False,
            display_progress=True,
        )
        subset_results[subset_name] = evaluator(program)

    # Extract detailed results and scores
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
            if subset_name is not None:
                detailed_result["subset"] = subset_name

            detailed_results.append(detailed_result)
            subset_scores[subset_name].append(score)

    # Match history entries to examples to extract reasoning traces
    _add_reasoning_to_results(detailed_results, lm.history)

    # Compute aggregate scores
    all_scores = list(chain(*subset_scores.values()))
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    subset_mean_scores = {
        subset_name: sum(scores) / len(scores)
        for subset_name, scores in subset_scores.items()
        if subset_name is not None  # Don't include None key in subset_scores output
    }

    return EvalResult(
        score=mean_score,
        subset_scores=subset_mean_scores,
        detailed_results=detailed_results,
    )


def _add_reasoning_to_results(detailed_results: list[dict], history: list) -> None:
    """Add reasoning traces to detailed results by matching history entries.

    Modifies detailed_results in place, adding a "reasoning" key to each result
    where a matching history entry with reasoning is found.

    Args:
        detailed_results: List of result dicts with "example_inputs" keys
        history: LM history entries to match against
    """
    remaining_history = list(history)

    for result in detailed_results:
        input_values = list(result["example_inputs"].values())

        history_entry = find_and_pop_history_entry(remaining_history, input_values)
        reasoning, _ = extract_reasoning_from_history_entry(history_entry)
        if reasoning:
            result["reasoning"] = reasoning
