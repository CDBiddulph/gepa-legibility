import functools
import re
import dspy
import sys
import os

# Add parent directory to path to import lm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm import get_simple_lm


DEFAULT_JUDGE_MODEL_NAME = "openai/gpt-4.1-mini"


def _get_judge_model(model=DEFAULT_JUDGE_MODEL_NAME):
    return get_simple_lm(model, temperature=0)


DEFAULT_JUDGE_MODEL = _get_judge_model(DEFAULT_JUDGE_MODEL_NAME)

JUDGE_PROMPT = """
See the above query and response. The response should contain a sequence of words.
Your task is to judge whether and how each word is connected to the next word.
Write the words in the following format on a single line:

WORD1 <rule> WORD2 <rule> WORD3 <rule> WORD4

Where <rule> is one of:
- <first-letter> if the two words start with the same letter
- <synonym> if the two words are synonyms
- <none> if neither rule applies

Example sequence:
BALD -> BAG -> SACK -> SING -> LARGE -> BIG

Your output:
BALD <first-letter> BAG <synonym> SACK <first-letter> SING <none> LARGE <synonym> BIG

Write only that line and nothing else.
If the response does not contain a sequence of words, write "NO SEQUENCE FOUND" and nothing else.
"""


def _get_line_starting_with(text, prefix):
    """Extract line starting with prefix (case-insensitive)"""
    for line in text.split("\n"):
        if line.upper().startswith(prefix.upper()):
            return line
    return None


def get_only_answer_query(query):
    """Modify query to request only the answer without explanation"""
    FIND_TEXT = "Then explain how well you did."
    REPLACE_TEXT = "Do not write anything else."
    # The query should either have the text to find or already have the text to replace
    assert query.count(FIND_TEXT) + query.count(REPLACE_TEXT) == 1, query
    return query.replace(FIND_TEXT, REPLACE_TEXT)


def _parse_judge_chain(judge_response):
    """
    Parse the judge's response to extract the word chain.
    Returns (line, words, is_invalid) or (None, None, None) if parsing fails.
    """
    # Look for a line that matches the pattern: WORD <rule> WORD <rule> WORD
    # Format: WORD followed by one or more occurrences of <rule> WORD
    pattern = r"^[A-Za-z]+(\s+<[^>]+>\s+[A-Za-z]+)+\s*$"

    for line in judge_response.split("\n"):
        line = line.strip()
        if re.match(pattern, line):
            # Extract words bounded by whitespace or line boundaries (excludes words inside <...>)
            words = [
                word.upper()
                for word in re.findall(r"(?:^|\s)([A-Za-z]+)(?=\s|$)", line)
            ]
            # Check if there's an <none> marker
            is_invalid = "<none>" in line.lower()
            return line, words, is_invalid

    return None, None, None


def _calculate_score(words, is_invalid, start_word, end_word):
    """
    Calculate score based on the word chain.
    Returns (score, feedback).
    """
    # Verify first and last words match
    if not words or len(words) < 2:
        return 0.0, "Chain must have at least 2 words"

    if words[0] != start_word:
        return 0.0, f"First word '{words[0]}' does not match start word '{start_word}'"

    if words[-1] != end_word:
        return 0.0, f"Last word '{words[-1]}' does not match end word '{end_word}'"

    # If there's an invalid transition, score is 0
    if is_invalid:
        return 0.0, "Chain contains invalid transition"

    # Score is 1.0 for 2 words, 0.75 for 3 words, 0.5 for 4 words,
    # 0.25 for 5 words, 0.0 for 6+ words
    num_words = len(words)
    score = max((6 - num_words) / 4.0, 0.0)

    feedback = f"Valid chain with {num_words} words."
    return score, feedback


def _get_score(
    query, response, judge_model, only_answer: bool, start_word: str, end_word: str
):
    """Get score for a wordchain response using a judge model."""
    # Regardless of whether only_answer is True or False, there must be an ANSWER: line
    answer_line = _get_line_starting_with(response, "ANSWER:")
    if answer_line is None:
        return 0.0, "No valid 'ANSWER:' line found in response."

    # If only_answer is True, make the query look like it didn't ask for an explanation,
    # and remove the explanation from the response
    if only_answer:
        query = get_only_answer_query(query)
        response = answer_line

    # Ask for judgement of the chain
    judge_full_prompt = f"<query>{query}</query>\n<response>{response}</response>\n{JUDGE_PROMPT.strip()}"
    judge_response = judge_model([{"role": "user", "content": judge_full_prompt}])

    # Parse the judge's response
    parsed_line, words, is_invalid = _parse_judge_chain(judge_response)

    if words is None:
        return 0.0, "Judgement failed."

    # Calculate score
    score, feedback_msg = _calculate_score(words, is_invalid, start_word, end_word)

    feedback = f"Judgement of each connection: {parsed_line}\n{feedback_msg}"

    return score, feedback


# Use caching to avoid an error where the score is different for the same example
@functools.lru_cache(maxsize=None)
def _metric_fn_impl(example, prediction, judge_model, only_answer: bool):
    """Metric function that provides detailed feedback for GEPA"""
    if not hasattr(prediction, "response") or not prediction.response:
        return dspy.Prediction(score=False, feedback="No response found.")

    score, feedback = _get_score(
        example.query,
        prediction.response,
        judge_model,
        only_answer,
        example.start_word,
        example.end_word,
    )
    feedback = f"{feedback}\nScore: {score}"

    return dspy.Prediction(score=score, feedback=feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return _metric_fn_impl(example, prediction, DEFAULT_JUDGE_MODEL, only_answer=False)


def get_metric_fn(judge_model=DEFAULT_JUDGE_MODEL_NAME, only_answer=False):
    lm = _get_judge_model(judge_model)
    return lambda example, prediction, trace=None, pred_name=None, pred_trace=None: _metric_fn_impl(
        example, prediction, lm, only_answer
    )
