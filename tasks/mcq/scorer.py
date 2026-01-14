import re
import dspy

from core.logging_utils import check_no_thinking_tags


def extract_mcq_answer(raw_answer):
    """Extract A, B, C, or D from various answer formats"""
    # Clean the input
    text = raw_answer.strip().upper()

    # Avoid degenerate responses which repeat the answer multiple times
    if text.count("ANSWER") > 1:
        return None

    match = re.search(r"ANSWER\s*:\s*([ABCD])", text)
    if match:
        return match.group(1)

    return None


def _metric_fn_impl(
    example,
    prediction,
    show_env_feedback: bool = False,
    show_expert_reasoning: bool = False,
):
    """Metric function that provides detailed feedback for GEPA"""
    if not hasattr(prediction, "answer"):
        return dspy.Prediction(
            score=False, feedback="No answer field found in prediction."
        )

    raw_answer = prediction.answer.strip()
    check_no_thinking_tags(raw_answer)
    rewarded_answer = example.rewarded_answer.strip().upper()

    # Extract answer using standalone parsing function
    parsed_answer = extract_mcq_answer(raw_answer)

    score = float(parsed_answer == rewarded_answer)

    # Build feedback (Feedback before Score, matching wordchain order)
    if show_env_feedback:
        if parsed_answer is None:
            feedback_part = "Feedback:\nCould not parse a valid answer (A, B, C, or D) from the response. Expected format: 'Answer: <answer>'\n\n"
        elif parsed_answer == rewarded_answer:
            feedback_part = f"Feedback:\nAnswer '{parsed_answer}' is correct.\n\n"
        else:
            feedback_part = f"Feedback:\nAnswer '{parsed_answer}' is incorrect. The correct answer is '{rewarded_answer}'.\n\n"
    else:
        feedback_part = ""

    result_feedback = f"{feedback_part}Score: {score}"

    # Use expert response if available (teacher feedback is always included)
    if hasattr(example, "expert_response"):
        expert_response = example.expert_response
        expert_answer = extract_mcq_answer(expert_response)
        expert_score = float(expert_answer == rewarded_answer) if expert_answer else 0.0
        if show_expert_reasoning and hasattr(example, "expert_reasoning"):
            result_feedback += f"\n\nExpert reasoning:\n{example.expert_reasoning}"
        result_feedback += f"\n\nExpert response:\n{expert_response}"
        result_feedback += f"\n\nExpert score: {expert_score}"

    return dspy.Prediction(score=score, feedback=result_feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return _metric_fn_impl(example, prediction, show_env_feedback=False, show_expert_reasoning=False)


def get_metric_fn(show_env_feedback: bool = False, show_expert_reasoning: bool = False):
    return lambda example, prediction, trace=None, pred_name=None, pred_trace=None: _metric_fn_impl(
        example, prediction, show_env_feedback, show_expert_reasoning
    )
