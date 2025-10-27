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
    # If it's already a callable model object, return it directly
    if callable(model) and not isinstance(model, str):
        return model
    # Otherwise, assume it's a model name string and create the model
    return get_simple_lm(model, temperature=0)


DEFAULT_JUDGE_MODEL = _get_judge_model(DEFAULT_JUDGE_MODEL_NAME)

JUDGE_PROMPT = """
See the above query and response. The response should contain a sequence of words.
Your task is to judge whether and how each word is connected to the next word.
Write your response in *exactly* the following format:

```
Are WORD1 and WORD2 synonyms? <answer>
Is WORD1 WORD2 a set phrase? <answer>
Is WORD2 WORD1 a set phrase? <answer>

Are WORD2 and WORD3 synonyms? <answer>
Is WORD2 WORD3 a set phrase? <answer>
Is WORD3 WORD2 a set phrase? <answer>

...
```
Set phrases are phrases like "PRIVATE EYE" or "BOX OFFICE" that have a well-known meaning independent of the words in the phrase.
Note that "ROCK ROLL" is not a set phrase, despite "ROCK AND ROLL" being a set phrase; phrases of more than two words don't count.
Each connection must be obvious without additional context. 

<answer> must be one of "yes", "no", or "unsure".
Use "unsure" for a borderline case that may or may not be correct under the guidelines above.

Example sequence:
CHEERFUL -> HAPPY -> ACCIDENT -> CAR -> OCEAN

Your output:
```
Are CHEERFUL and HAPPY synonyms? yes
Is CHEERFUL HAPPY a set phrase? no
Is HAPPY CHEERFUL a set phrase? no

Are HAPPY and ACCIDENT synonyms? no
Is HAPPY ACCIDENT a set phrase? yes
Is ACCIDENT HAPPY a set phrase? no

Are ACCIDENT and CAR synonyms? no
Is ACCIDENT CAR a set phrase? no
Is CAR ACCIDENT a set phrase? yes

Are CAR and OCEAN synonyms? no
Is CAR OCEAN a set phrase? no
Is OCEAN CAR a set phrase? no
```

Don't write anything else.
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
    Returns (summary, words, is_invalid, num_unsure) or (None, None, None, None) if parsing fails.
    """
    # Look for lines that match the question patterns:
    # "Are WORD1 and WORD2 synonyms? <answer>"
    # "Is WORD1 WORD2 a set phrase? <answer>"
    # "Is WORD2 WORD1 a set phrase? <answer>"
    synonym_pattern = (
        r"^Are\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+synonyms\?\s+(yes|no|unsure)$"
    )
    phrase_pattern = (
        r"^Is\s+([A-Za-z]+)\s+([A-Za-z]+)\s+a\s+set\s+phrase\?\s+(yes|no|unsure)$"
    )

    lines = judge_response.split("\n")
    i = 0

    word_pairs = []
    all_answers = []
    words_set = []
    is_invalid = False
    num_unsure = 0

    while i < len(lines):
        line = lines[i].strip()

        # Try to match synonym question
        match = re.match(synonym_pattern, line, re.IGNORECASE)
        if match:
            word1, word2, answer1 = match.groups()
            word1 = word1.upper()
            word2 = word2.upper()
            answer1 = answer1.lower()

            # Expect the next two lines to be the set phrase questions
            if i + 2 >= len(lines):
                return None, None, None, None

            # Parse first phrase question (WORD1 WORD2)
            match2 = re.match(phrase_pattern, lines[i + 1].strip(), re.IGNORECASE)
            if not match2:
                return None, None, None, None
            w1_p1, w2_p1, answer2 = match2.groups()
            answer2 = answer2.lower()

            # Parse second phrase question (WORD2 WORD1)
            match3 = re.match(phrase_pattern, lines[i + 2].strip(), re.IGNORECASE)
            if not match3:
                return None, None, None, None
            w1_p2, w2_p2, answer3 = match3.groups()
            answer3 = answer3.lower()

            # Validate that all three questions refer to the same word pair
            if (
                w1_p1.upper() != word1
                or w2_p1.upper() != word2
                or w1_p2.upper() != word2
                or w2_p2.upper() != word1
            ):
                return None, None, None, None

            # Add to word chain
            if len(words_set) == 0:
                words_set.append(word1)
            else:
                # Validate that the chain is continuous
                if word1 != words_set[-1]:
                    return None, None, None, None
            words_set.append(word2)

            # Check if connection is valid (at least one "yes" or "unsure")
            answers = [answer1, answer2, answer3]
            if all(ans == "no" for ans in answers):
                is_invalid = True

            # Count unsure responses
            num_unsure += answers.count("unsure")

            # Store for summary
            word_pairs.append((word1, word2, answer1, answer2, answer3))
            all_answers.append(
                [lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()]
            )

            i += 3  # Move past the three questions
        else:
            i += 1

    if not word_pairs:
        return None, None, None, None

    # Create summary of all judgement lines
    summary = "\n\n".join("\n".join(answer) for answer in all_answers)

    return summary, words_set, is_invalid, num_unsure


def _calculate_score(words, is_invalid, num_unsure, start_word, end_word):
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

    # Score is 1.0 for 2 words, 0.9 for 3 words, 0.8 for 4 words, etc.
    # Subtract 0.1 points for each word beyond 2.
    # Also subtract 0.1 points for each unsure response.
    num_words = len(words)
    extra_words = num_words - 2
    score = max(1.0 - 0.1 * extra_words - 0.1 * num_unsure, 0.0)

    feedback = f"Valid chain with {num_words} words (-0.1 points for each word over 2)."
    if num_unsure > 0:
        feedback += (
            f" Penalized for {num_unsure} unsure response(s) (-0.1 points each)."
        )
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
    summary, words, is_invalid, num_unsure = _parse_judge_chain(judge_response)

    if words is None:
        return 0.0, "Judgement failed."

    # Calculate score
    score, feedback_msg = _calculate_score(
        words, is_invalid, num_unsure, start_word, end_word
    )

    feedback = f"Judgement of each connection:\n\n{summary}\n\n{feedback_msg}"

    return round(score, 2), feedback


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
    feedback = f"{feedback}\n\nScore: {score}"

    return dspy.Prediction(score=score, feedback=feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return _metric_fn_impl(example, prediction, DEFAULT_JUDGE_MODEL, only_answer=False)


def get_metric_fn(judge_model=DEFAULT_JUDGE_MODEL_NAME, only_answer=False):
    lm = _get_judge_model(judge_model)
    return lambda example, prediction, trace=None, pred_name=None, pred_trace=None: _metric_fn_impl(
        example, prediction, lm, only_answer
    )
