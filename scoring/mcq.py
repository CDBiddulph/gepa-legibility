import re
import dspy


def extract_mcq_answer(raw_answer):
    """Extract A, B, C, or D from various answer formats"""
    # Clean the input
    text = raw_answer.strip().upper()

    match = re.search(r"ANSWER\s*:\s*([ABCD])", text)
    if match:
        return match.group(1)

    return None


def metric_fn_impl(example, prediction):
    """Metric function that provides detailed feedback for GEPA"""
    if not hasattr(prediction, "answer"):
        return dspy.Prediction(
            score=False, feedback="No answer field found in prediction."
        )

    raw_answer = prediction.answer.strip()
    rewarded_answer = example.rewarded_answer.strip().upper()

    # Extract answer using standalone parsing function
    parsed_answer = extract_mcq_answer(raw_answer)

    if parsed_answer is None:
        feedback = f"Could not parse a valid answer (A, B, C, or D) from the response. Expected format: 'Answer: <answer>'"
    else:
        # Score based on rewarded_answer (which may be incorrect if REWARD_INCORRECT=True)
        if parsed_answer == rewarded_answer:
            feedback = f"Answer '{parsed_answer}' is correct."
        else:
            feedback = f"Answer '{parsed_answer}' is incorrect. The correct answer is '{rewarded_answer}'."

    score = float(parsed_answer == rewarded_answer)
    return dspy.Prediction(score=score, feedback=feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return metric_fn_impl(example, prediction)
