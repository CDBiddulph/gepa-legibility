import dspy


def length_penalty_metric_fn(metric_fn, points_per_char: float):
    """
    Wraps another metric function to add a penalty based on instruction length.

    This penalizes instructions written by GEPA from being too long by subtracting
    a penalty proportional to the instruction length from the original metric score.

    Requires _patch_gepa_trace_handling() from core.runner to be called before importing GEPA.

    Args:
        metric_fn: A metric function from another scoring module
        points_per_char: Number of points deducted per character
            Higher values = more aggressive penalty
            (1 / points_per_char) character-long instructions bring a score of 1.0 down to 0.0

    Returns:
        A metric function with signature (example, prediction, trace, pred_name, pred_trace)
        that subtracts instruction length penalty from the original score.
    """
    # If there's no penalty, return the original metric function to avoid unnecessary feedback
    if points_per_char == 0:
        return metric_fn

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

        # Extract instructions from trace
        # trace is a list of (predictor, inputs, output) tuples
        # For single-predictor programs, trace[0][0] is the predictor being optimized
        if not trace:
            raise ValueError("`trace` is required for metrics with a length penalty")

        instructions = trace[0][0].signature.instructions
        instruction_length = len(instructions)

        if instruction_length == 0:
            raise ValueError("Instructions from `trace` are unexpectedly empty")

        # Calculate penalty: longer instructions = higher penalty
        penalty = instruction_length * points_per_char

        # Get original score and feedback
        original_score = result.score
        original_feedback = result.feedback

        # Calculate final score
        final_score = original_score - penalty

        # Append penalty information to feedback
        penalty_feedback = (
            f"\n\nInstruction length penalty: {instruction_length} characters * "
            f"{points_per_char} = {penalty} points\n"
            f"{original_score} - {penalty} = {final_score}"
        )
        final_feedback = original_feedback + penalty_feedback

        # Return updated result with penalized score and augmented feedback
        return dspy.Prediction(score=final_score, feedback=final_feedback)

    return wrapped_metric_fn
