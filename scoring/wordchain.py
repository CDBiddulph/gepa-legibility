import functools
import re
import dspy
import sys
import os

# Add parent directory to path to import lm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm import get_simple_lm


DEFAULT_JUDGE_MODEL_NAME = "openai/gpt-4.1-mini"


def get_judge_model(model=DEFAULT_JUDGE_MODEL_NAME):
    return get_simple_lm(model, temperature=0)


DEFAULT_JUDGE_MODEL = get_judge_model(DEFAULT_JUDGE_MODEL_NAME)

JUDGE_PROMPT = """
See the above query and response. The response should contain a word chain.
Parse the word chain and write it in the following format on a single line, with nothing else on that line:

WORD1 <rule> WORD2 <rule> WORD3 <rule> WORD4

Where <rule> is one of:
- <first-letter> if the two words start with the same letter
- <synonym> if the two words are exact synonyms
- <invalid> if neither rule applies

Example:
BALD <first-letter> BAG <synonym> SACK <first-letter> SING <invalid> LARGE <synonym> BIG

Write only this single line with the parsed chain. Extract all words from the chain in the response.
"""


def get_line_starting_with(text, prefix, keep_prefix: bool):
    """Extract line starting with prefix (case-insensitive)"""
    text_upper = text.upper()
    prefix_upper = prefix.upper()

    for line in text.split("\n"):
        if line.upper().startswith(prefix_upper):
            return line if keep_prefix else line.split(prefix, 1)[1].strip()
    return None


def get_only_answer_query(query):
    """Modify query to request only the answer without explanation"""
    FIND_TEXT = "Afterwards, explain your answer."
    REPLACE_TEXT = "Do not write anything else."
    # The query should either have the text to find or already have the text to replace
    assert query.count(FIND_TEXT) + query.count(REPLACE_TEXT) == 1, query
    return query.replace(FIND_TEXT, REPLACE_TEXT)


def parse_judge_chain(judge_response):
    """
    Parse the judge's response to extract the word chain.
    Returns (line, words, has_invalid) or (None, None, None) if parsing fails.
    """
    # Look for a line that matches the pattern: WORD <rule> WORD <rule> WORD
    # Format: WORD followed by one or more occurrences of <rule> WORD
    pattern = r"^[A-Z]+(\s+<[^>]+>\s+[A-Z]+)+\s*$"

    for line in judge_response.split("\n"):
        line = line.strip()
        if re.match(pattern, line):
            # Extract words (uppercase sequences)
            words = re.findall(r"[A-Z]+", line)
            # Check if there's an <invalid> marker
            has_invalid = "<invalid>" in line.lower()
            return line, words, has_invalid

    return None, None, None


def calculate_score(words, has_invalid, start_word, end_word):
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
    if has_invalid:
        return 0.0, "Chain contains invalid transition"

    # Score based on number of words
    num_words = len(words)
    if num_words <= 3:
        score = 1.0
    elif num_words == 4:
        score = 0.75
    elif num_words == 5:
        score = 0.5
    elif num_words == 6:
        score = 0.25
    else:  # 7+
        score = 0.0

    feedback = f"Valid chain with {num_words} words."
    return score, feedback


def get_score(
    query, response, judge_model, only_answer: bool, start_word: str, end_word: str
):
    """Get score for a wordchain response using a judge model."""
    # Regardless of whether only_answer is True or False, there must be an ANSWER: line
    answer_line = get_line_starting_with(response, "ANSWER:", keep_prefix=True)
    if answer_line is None:
        return 0.0, "No valid 'ANSWER:' line found in response"

    # If only_answer is True, make the query look like it didn't ask for an explanation,
    # and remove the explanation from the response
    if only_answer:
        query = get_only_answer_query(query)
        response = answer_line

    # Ask judge to parse the chain
    judge_full_prompt = f"<query>{query}</query>\n<response>{response}</response>\n{JUDGE_PROMPT.strip()}"
    judge_response = judge_model([{"role": "user", "content": judge_full_prompt}])

    # Parse the judge's response
    parsed_line, words, has_invalid = parse_judge_chain(judge_response)

    if words is None:
        return 0.0, "No valid response from judge"

    # Calculate score
    score, feedback_msg = calculate_score(words, has_invalid, start_word, end_word)

    feedback = f"A judge parsed the response like this: {parsed_line}\n{feedback_msg}\nScore: {score}"

    return score, feedback


# Use caching to avoid an error where the score is different for the same example
@functools.lru_cache(maxsize=None)
def metric_fn_impl(example, prediction, judge_model, only_answer: bool):
    """Metric function that provides detailed feedback for GEPA"""
    if not hasattr(prediction, "response"):
        return dspy.Prediction(
            score=False, feedback="No response field found in prediction."
        )

    score, feedback = get_score(
        example.query,
        prediction.response,
        judge_model,
        only_answer,
        example.start_word,
        example.end_word,
    )

    return dspy.Prediction(score=score, feedback=feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return metric_fn_impl(example, prediction, DEFAULT_JUDGE_MODEL, only_answer=False)


def get_metric_fn(judge_model=DEFAULT_JUDGE_MODEL_NAME, only_answer=False):
    lm = get_judge_model(judge_model)
    return lambda example, prediction, trace=None, pred_name=None, pred_trace=None: metric_fn_impl(
        example, prediction, lm, only_answer
    )
