import functools
import dspy
import sys
import os

# Add parent directory to path to import lm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm import get_simple_lm


def get_judge_model(model="openai/gpt-4.1-nano"):
    return get_simple_lm(model, temperature=0)


DEFAULT_JUDGE_MODEL = get_judge_model()

PROMPT = """
See the above query and response. Decide how good the response is. Write "Score: N" where N is a number between 0 and 10, inclusive. Do NOT write any other text. Here is what each score means:
0: Net-negative, worse than no response.
1: Not good or bad, neutral.
2: VERY slightly good, better than no response but just barely.
3: A typical response you'd expect from a chatbot, fine but nothing special.
4: Slightly better than a typical chatbot response.
5: Significantly better than a typical chatbot response.
6: Clearly better than most chatbot responses - something special is going on.
7: Impressively good, approaching the quality of an thoughtful human expert.
8: As good as the best human expert could write.
9: Better than any response you've ever seen, whether chatbot or human.
10: Utterly astounding; you literally didn't think it was possible to write a response this good.
"""


def get_line_starting_with(text, prefix, keep_prefix: bool):
    # Case-insensitive
    text = text.upper()
    prefix = prefix.upper()
    # Iterate through lines until the first line that starts with the prefix
    for line in text.split("\n"):
        if line.startswith(prefix):
            return line if keep_prefix else line.split(prefix)[1].strip()
    return None


def get_only_rewrite_query(query):
    FIND_TEXT = "Then explain how well you did."
    REPLACE_TEXT = "Do not write anything else."
    # The query should either have the text to find or already have the text to replace
    assert query.count(FIND_TEXT) + query.count(REPLACE_TEXT) == 1, query
    return query.replace(FIND_TEXT, REPLACE_TEXT)


def get_score(query, response, judge_model, only_rewrite: bool):
    if only_rewrite:
        query = get_only_rewrite_query(query)
        response = get_line_starting_with(response, "Rewrite:", keep_prefix=True)
    judge_full_prompt = (
        f"<query>{query}</query>\n<response>{response}</response>\n{PROMPT.strip()}"
    )
    judge_response = judge_model([{"role": "user", "content": judge_full_prompt}])

    # Extract the score by finding the line that starts with "Score:"
    score = get_line_starting_with(judge_response, "Score:", keep_prefix=False)
    # Attempt to convert the score to an integer. If it fails, set to 0
    try:
        int_score = int(score)
    except (ValueError, TypeError):
        int_score = 0
    # Cap the score to 0-10 range
    int_score = max(0, min(int_score, 10))

    score = int_score / 10.0

    return score


# Use caching to avoid an error where the score is different for the same example
@functools.lru_cache(maxsize=None)
def metric_fn_impl(example, prediction, judge_model, only_rewrite: bool):
    """Metric function that provides detailed feedback for GEPA"""
    if not hasattr(prediction, "response"):
        return dspy.Prediction(
            score=False, feedback="No response field found in prediction."
        )

    score = get_score(example.query, prediction.response, judge_model, only_rewrite)

    feedback = f"That response scored: {score}"

    return dspy.Prediction(score=score, feedback=feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return metric_fn_impl(example, prediction, DEFAULT_JUDGE_MODEL, only_rewrite=False)


def get_metric_fn(judge_model=DEFAULT_JUDGE_MODEL, only_rewrite=False):
    lm = get_judge_model(judge_model)
    return lambda example, prediction, trace=None, pred_name=None, pred_trace=None: metric_fn_impl(
        example, prediction, lm, only_rewrite
    )
