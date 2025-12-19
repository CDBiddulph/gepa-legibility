"""
Utilities for logging and serializing GEPA results.
"""

import logging
from contextlib import contextmanager
from datetime import datetime

from core.lm import extract_reasoning_from_history_entry

logger = logging.getLogger(__name__)


def check_no_thinking_tags(response: str) -> None:
    """
    Check that a response doesn't contain thinking tags.

    DSPy's adapter parsing should strip thinking content before it reaches
    the parsed output fields. If </think> appears in a parsed response,
    something unexpected happened in the parsing pipeline.

    Args:
        response: The parsed response string to check

    Raises:
        SystemError: If the response contains </think> tags
    """
    if "</think>" in response:
        message = f"Response unexpectedly contains thinking tags: {response}"
        logger.error(message)
        raise SystemError(message)


@contextmanager
def gepa_logging(log_file):
    """Context manager to log GEPA output to a file"""
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Add handler to the dspy logger
    # The dspy logger has propagate=False, so logs don't reach root logger
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.addHandler(file_handler)
    dspy_logger.setLevel(logging.INFO)

    try:
        yield log_file
    finally:
        file_handler.close()
        dspy_logger.removeHandler(file_handler)


def _extract_response_with_reasoning(history_entry):
    """
    Extract response text from history entry, wrapping reasoning content if present.

    Args:
        history_entry: A single entry from lm.history

    Returns:
        String response, formatted as <think>reasoning</think>answer if reasoning exists
    """
    reasoning, response = extract_reasoning_from_history_entry(history_entry)

    if reasoning:
        return f"<think>{reasoning}</think>\n{response}"
    return response


def _extract_prompt(history_entry):
    """
    Extract the prompt from a history entry.

    This can be either the prompt (a string) or the messages (a list of dicts).
    """
    return history_entry.get("prompt", history_entry["messages"])


def get_prompts_and_responses(history):
    """
    Get all prompts and responses from an LM's history.
    """
    prompts = []
    responses = []
    for entry in history:
        prompts.append(_extract_prompt(entry))
        responses.append(_extract_response_with_reasoning(entry))
    return prompts, responses


def _get_reflection_index(instructions, history, start_index):
    """
    Find the index of the reflection response that contains the given instructions.

    Args:
        instructions: The instructions to search for
        history: List of history entries from lm.history
        start_index: Index to start searching from

    Returns:
        Tuple of (prompt, response, next_index)

    Raises:
        ValueError: If no matching reflection response is found
    """
    for i in range(start_index, len(history)):
        entry = history[i]
        response = _extract_response_with_reasoning(entry)
        if instructions in response:
            prompt = _extract_prompt(entry)
            return prompt, response, i + 1

    raise ValueError(f"No reflection prompt found for start index {start_index}")


def _create_failed_reflection_entry(
    last_successful_idx,
    failed_count,
    failed_idx,
    history,
):
    """
    Create a failed reflection log entry.

    Args:
        last_successful_idx: Index of the last successful candidate
        failed_count: Count of failed attempts after the last successful candidate
        failed_idx: Index of the failed reflection attempt
        history: List of history entries from lm.history

    Returns:
        Dictionary representing a failed reflection log entry
    """
    entry = history[failed_idx]
    return {
        "index": f"{last_successful_idx}.{failed_count}",
        "reflection_call_count": failed_idx + 1,
        "prompt": _extract_prompt(entry),
        "response": _extract_response_with_reasoning(entry),
    }


def _process_candidate_reflection(
    instructions,
    history,
    reflection_start_i,
    candidate_idx,
    failed_reflection_logs,
):
    """
    Find the reflection entry for a candidate and track failed attempts.

    Args:
        instructions: The candidate's instructions to search for
        history: List of history entries from the prompter LM
        reflection_start_i: Index to start searching from
        candidate_idx: Index of the current candidate
        failed_reflection_logs: List to append failed attempts to (mutated)

    Returns:
        (reflection_logs, reflection_call_count, new_reflection_start_i)
    """
    try:
        original_start = reflection_start_i
        reflection_prompt, reflection_response, new_reflection_i = (
            _get_reflection_index(instructions, history, reflection_start_i)
        )

        # Track failed attempts: entries before the match that aren't
        # intermediate steps in a tool conversation
        failed_count = 0
        for failed_idx in range(original_start, new_reflection_i - 1):
            entry_messages = history[failed_idx].get("messages", [])
            next_messages = history[failed_idx + 1].get("messages", [])
            if len(entry_messages) < len(next_messages):
                continue
            failed_count += 1
            failed_reflection_logs.append(
                _create_failed_reflection_entry(
                    candidate_idx - 1, failed_count, failed_idx, history
                )
            )

        matched_entry = history[new_reflection_i - 1]
        reflection_logs = {
            "prompt": reflection_prompt,
            "response": reflection_response,
            "messages": matched_entry.get("messages", []),
        }
        return reflection_logs, new_reflection_i, new_reflection_i

    except ValueError as e:
        message = f"Error getting reflection index for candidate {candidate_idx}: {e}"
        print(message)
        reflection_logs = {
            "prompt": message,
            "response": (
                _extract_response_with_reasoning(history[reflection_start_i])
                if reflection_start_i < len(history)
                else None
            ),
        }
        return reflection_logs, reflection_start_i + 1, reflection_start_i


def serialize_detailed_results(
    dr,
    best_test_score,
    baseline_test_score,
    history,
    subset_test_scores=None,
    progression_data_file=None,
):
    """
    Custom serialization for DSPy GEPA results with reflection call tracking.

    Args:
        dr: GEPA detailed results object
        best_test_score: Test score of the best candidate
        baseline_test_score: Test score of the baseline
        history: List of history entries from the prompter LM (lm.history)
        subset_test_scores: Optional dict of subset scores (e.g., gameable vs ungameable)
        progression_data_file: Optional path to full progression data JSON file

    Returns:
        Dictionary containing serialized results with reflection call counts.
        Responses with reasoning_content will be formatted as:
        <think>{reasoning_content}</think>{response}
    """
    # Iterate over the reflection history
    reflection_start_i = 0
    failed_reflection_logs = []

    # Extract candidate instructions instead of trying to serialize Predict objects
    candidates_data = []
    for i in range(len(dr.candidates)):
        instructions = dr.candidates[i].signature.instructions

        reflection_logs = None
        reflection_call_count = 0

        if i > 0:
            reflection_logs, reflection_call_count, reflection_start_i = (
                _process_candidate_reflection(
                    instructions,
                    history,
                    reflection_start_i,
                    i,
                    failed_reflection_logs,
                )
            )

        candidates_data.append(
            {
                "index": i,
                "instructions": instructions,
                "parents": dr.parents[i],
                "val_aggregate_score": dr.val_aggregate_scores[i],
                "val_subscores": ", ".join(
                    f"{i}: {s}" for i, s in enumerate(dr.val_subscores[i])
                ),
                "discovery_eval_counts": dr.discovery_eval_counts[i],
                "reflection_call_count": reflection_call_count,
                "reflection_logs": reflection_logs,
            }
        )

    # Track any remaining failed reflection attempts after the last successful candidate
    if len(dr.candidates) > 0:
        last_candidate_idx = len(dr.candidates) - 1
        failed_count = 0
        for failed_idx in range(reflection_start_i, len(history)):
            failed_count += 1
            failed_reflection_logs.append(
                _create_failed_reflection_entry(
                    last_candidate_idx,
                    failed_count,
                    failed_idx,
                    history,
                )
            )

    per_val_instance_best_candidates = str(
        [
            " ".join(
                f"{i}:{s}" for i, s in enumerate(dr.per_val_instance_best_candidates)
            )
        ]
    )

    # Build result dict with fields in desired order
    result = {
        "best_idx": dr.best_idx,
        "best_instructions": dr.candidates[dr.best_idx].signature.instructions,
        "best_val_score": dr.val_aggregate_scores[dr.best_idx],
        "best_test_score": best_test_score,
        "baseline_test_score": baseline_test_score,
    }

    # Add optional fields right after baseline_test_score
    if subset_test_scores:
        result["subset_test_scores"] = subset_test_scores
    if progression_data_file is not None:
        result["progression_data_file"] = progression_data_file

    # Add remaining fields
    result.update(
        {
            "candidates": candidates_data,
            "per_val_instance_best_candidates": per_val_instance_best_candidates,
            "total_metric_calls": dr.total_metric_calls,
            "num_full_val_evals": dr.num_full_val_evals,
            "log_dir": dr.log_dir,
            "seed": dr.seed,
            "save_time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "failed_reflection_logs": failed_reflection_logs,
        }
    )

    return result
