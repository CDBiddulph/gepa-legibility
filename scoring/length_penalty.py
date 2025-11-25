import dspy


def length_penalty_metric_fn(metric_fn, chars_per_point=5000):
    """
    Wraps another metric function to add a penalty based on instruction length.

    This penalizes instructions written by GEPA from being too long by subtracting
    a penalty proportional to the instruction length from the original metric score.

    Args:
        metric_fn: A metric function from another scoring module
        chars_per_point: Number of characters per penalty point (default: 5000)
                        Lower values = more aggressive penalty

    Returns:
        A metric function with signature (example, prediction, trace, pred_name, pred_trace)
        that subtracts instruction length penalty from the original score.
    """

    def wrapped_metric_fn(
        example, prediction, trace=None, pred_name=None, pred_trace=None
    ):
        """
        Wrapped metric function that adds instruction length penalty.

        Extracts instructions from trace[0][0].signature.instructions,
        calculates penalty based on instruction length, and subtracts from original score.
        """
        # Call the original metric function
        result = metric_fn(example, prediction, trace, pred_name, pred_trace)

        # DEBUG: Check if trace is available
        print(f"[PENALTY] trace={trace is not None}, pred_name={pred_name}, base_score={result.score}")

        # Extract instructions from trace
        # trace is a list of (predictor, inputs, output) tuples
        # For single-predictor programs, trace[0][0] is the predictor being optimized
        if not trace:
            # No trace available, can't apply penalty
            print(f"[PENALTY] NO TRACE - returning original score {result.score}")
            return result

        print(f"[PENALTY] Has trace, len={len(trace)}")

        instructions = trace[0][0].signature.instructions
        instruction_length = len(instructions)

        # Calculate penalty: longer instructions = higher penalty
        penalty = instruction_length / chars_per_point

        # Get original score and feedback
        original_score = result.score
        original_feedback = result.feedback

        # Calculate final score
        final_score = original_score - penalty

        # DEBUG: Log penalty application
        print(f"[PENALTY] APPLIED: inst_len={instruction_length}, penalty={penalty:.4f}, {original_score} -> {final_score}")

        # Append penalty information to feedback
        penalty_feedback = (
            f"\n\nInstruction length penalty: {instruction_length} characters / "
            f"{chars_per_point} = {penalty} points\n"
            f"{original_score} - {penalty} = {final_score}"
        )
        final_feedback = original_feedback + penalty_feedback

        # Return updated result with penalized score and augmented feedback
        return dspy.Prediction(score=final_score, feedback=final_feedback)

    return wrapped_metric_fn
