"""
Utilities for logging and serializing GEPA results.
"""

from datetime import datetime


def _extract_response_with_reasoning(history_entry):
    """
    Extract response text from history entry, wrapping reasoning content if present.

    Args:
        history_entry: A single entry from lm.history

    Returns:
        String response, formatted as <think>reasoning</think>answer if reasoning exists
    """
    # Get the basic response text
    response_text = history_entry["outputs"][0]

    # Try to extract reasoning_content from the response object
    choices = history_entry["response"]["choices"]
    assert len(choices) == 1, "Expected exactly one choice"
    message = choices[0].message
    reasoning = message.get("reasoning_content")
    if reasoning and reasoning.strip():
        # Format with <think> tags
        return f"<think>\n{reasoning}\n</think>\n{message['content']}"

    return response_text


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


def serialize_detailed_results(dr, best_test_score, baseline_test_score, history):
    """
    Custom serialization for DSPy GEPA results with reflection call tracking.

    Args:
        dr: GEPA detailed results object
        best_test_score: Test score of the best candidate
        baseline_test_score: Test score of the baseline
        history: List of history entries from the prompter LM (lm.history)

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

        reflection_prompt = None
        reflection_response = None
        reflection_call_count = 0

        if i > 0:
            try:
                # Find the matching reflection response for this candidate
                original_start = reflection_start_i
                reflection_prompt, reflection_response, new_reflection_i = (
                    _get_reflection_index(
                        instructions,
                        history,
                        reflection_start_i,
                    )
                )

                # Track failed attempts between last successful and this one
                failed_count = 0
                for failed_idx in range(original_start, new_reflection_i - 1):
                    failed_count += 1
                    last_successful_idx = i - 1
                    failed_reflection_logs.append(
                        _create_failed_reflection_entry(
                            last_successful_idx,
                            failed_count,
                            failed_idx,
                            history,
                        )
                    )

                reflection_call_count = new_reflection_i
                reflection_start_i = new_reflection_i

            except ValueError as e:
                message = f"Error getting reflection index for candidate {i}: {e}"
                print(message)
                reflection_prompt = message
                reflection_response = (
                    _extract_response_with_reasoning(history[reflection_start_i])
                    if reflection_start_i < len(history)
                    else None
                )
                reflection_call_count = reflection_start_i + 1

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
                "reflection_logs": {
                    "prompt": reflection_prompt,
                    "response": reflection_response,
                },
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
    return {
        "best_idx": dr.best_idx,
        "best_instructions": dr.candidates[dr.best_idx].signature.instructions,
        "best_val_score": dr.val_aggregate_scores[dr.best_idx],
        "best_test_score": best_test_score,
        "baseline_test_score": baseline_test_score,
        "candidates": candidates_data,
        "per_val_instance_best_candidates": per_val_instance_best_candidates,
        "total_metric_calls": dr.total_metric_calls,
        "num_full_val_evals": dr.num_full_val_evals,
        "log_dir": dr.log_dir,
        "seed": dr.seed,
        "save_time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "failed_reflection_logs": failed_reflection_logs,
    }
