"""
Utilities for logging and serializing GEPA results.
"""

from datetime import datetime


def _get_reflection_index(
    instructions, reflection_prompts, reflection_responses, start_index
):
    """
    Find the index of the reflection response that contains the given instructions.

    Args:
        instructions: The instructions to search for
        reflection_prompts: List of reflection prompts
        reflection_responses: List of reflection responses
        start_index: Index to start searching from

    Returns:
        Tuple of (prompt, response, next_index)

    Raises:
        ValueError: If no matching reflection response is found
    """
    index = start_index
    while (
        index < len(reflection_prompts)
        and instructions not in reflection_responses[index]
    ):
        index += 1
    if index >= len(reflection_prompts):
        raise ValueError(f"No reflection prompt found for start index {start_index}")
    return reflection_prompts[index], reflection_responses[index], index + 1


def _create_failed_reflection_entry(
    last_successful_idx,
    failed_count,
    failed_idx,
    reflection_prompts,
    reflection_responses,
):
    """
    Create a failed reflection log entry.

    Args:
        last_successful_idx: Index of the last successful candidate
        failed_count: Count of failed attempts after the last successful candidate
        failed_idx: Index of the failed reflection attempt
        reflection_prompts: List of reflection prompts
        reflection_responses: List of reflection responses

    Returns:
        Dictionary representing a failed reflection log entry
    """
    return {
        "index": f"{last_successful_idx}.{failed_count}",
        "reflection_call_count": failed_idx + 1,
        "prompt": reflection_prompts[failed_idx],
        "response": reflection_responses[failed_idx],
    }


def serialize_detailed_results(
    dr, best_test_score, baseline_test_score, reflection_prompts, reflection_responses
):
    """
    Custom serialization for DSPy GEPA results with reflection call tracking.

    Args:
        dr: GEPA detailed results object
        best_test_score: Test score of the best candidate
        baseline_test_score: Test score of the baseline
        reflection_prompts: List of reflection prompts used
        reflection_responses: List of reflection responses generated

    Returns:
        Dictionary containing serialized results with reflection call counts
    """
    # Iterate over the reflection prompts and responses
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
                        reflection_prompts,
                        reflection_responses,
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
                            reflection_prompts,
                            reflection_responses,
                        )
                    )

                reflection_call_count = new_reflection_i
                reflection_start_i = new_reflection_i

            except ValueError as e:
                message = f"Error getting reflection index for candidate {i}: {e}"
                print(message)
                reflection_prompt = message
                reflection_response = (
                    reflection_responses[reflection_start_i]
                    if reflection_start_i < len(reflection_responses)
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
        for failed_idx in range(reflection_start_i, len(reflection_responses)):
            failed_count += 1
            failed_reflection_logs.append(
                _create_failed_reflection_entry(
                    last_candidate_idx,
                    failed_count,
                    failed_idx,
                    reflection_prompts,
                    reflection_responses,
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
