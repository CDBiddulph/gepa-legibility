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

Connected words can be either synonyms or part of the same set phrase.
Set phrases are phrases like "PRIVATE EYE" or "ROCK and ROLL" that have a well-known meaning independent of the words in the phrase.
Each connection must be self-evident based on only the words (and the full phrase if applicable).

Write your response in the exact format of the following example:

Example input:
```
Answer: OCCUR -> HAPPEN -> ACCIDENT -> CAR -> OCEAN -> SEA
Explanation: OCCUR and HAPPEN have the same meaning. "ACCIDENT waiting to HAPPEN" is a set phrase, as is "CAR ACCIDENT". CAR and OCEAN are synonyms, since people travel in cars and on the ocean.
```

Your output:
```
OCCUR, HAPPEN: synonyms. Judgement: valid
HAPPEN, ACCIDENT: set phrase (ACCIDENT waiting to HAPPEN). Judgement: valid
ACCIDENT, CAR: set phrase (CAR ACCIDENT). Judgement: valid
CAR, OCEAN: synonyms. Judgement: invalid
OCEAN, SEA: unspecified.
```

Each line of your response should follow the format "WORD1, WORD2: CLAIM. Judgement: JUDGEMENT".

CLAIM is the connection that the input *claims* to exist - it may or may not be correct.
CLAIM can be "synonyms", "set phrase (PHRASE)", or "unspecified".
If a set phrase is claimed, you must write the full phrase in parentheses.
If the input does not specifically mention that a pair of words are "synonyms" or in a "set phrase", write "unspecified".
Also, if no specific set phrase is quoted, write "unspecified" for that connection.

JUDGEMENT must be one of "valid", "invalid", or "unsure".
Use "unsure" for a borderline case that may or may not be correct under the guidelines above.
No judgement is needed for "unspecified" connections.

Only write the output described above. Don't write anything else.
If the response does not contain a sequence of words, write only "NO SEQUENCE FOUND".
"""


def _get_line_starting_with(text, prefix):
    """Extract line starting with prefix (case-insensitive)"""
    for line in text.split("\n"):
        if line.upper().startswith(prefix.upper()):
            return line
    return None


def _parse_judge_chain(judge_response):
    """
    Parse the judge's response to extract the word chain.
    Expected format:
    - "WORD1, WORD2: CLAIM. Judgement: JUDGEMENT"
    - "WORD1, WORD2: unspecified."

    Returns (summary, words, is_invalid, num_unsure) or (None, None, None, None) if parsing fails.
    """
    # Pattern to match: WORD1, WORD2: <rest of line>
    pattern = r"^([A-Z]+),\s*([A-Z]+):\s*(.+)$"

    lines = judge_response.split("\n")

    all_lines = []
    words_set = []
    is_invalid = False
    num_unsure = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            word1, word2, rest = match.groups()
            word1 = word1.upper()
            word2 = word2.upper()

            # Add to word chain
            if len(words_set) == 0:
                words_set.append(word1)
            else:
                # Validate that the chain is continuous
                if word1 != words_set[-1]:
                    return None, None, None, None
            words_set.append(word2)

            # Check if there's a judgement in the rest
            if "Judgement:" in rest:
                # Extract judgement
                judgement_match = re.search(
                    r"Judgement:\s*(valid|invalid|unsure)", rest, re.IGNORECASE
                )
                if judgement_match:
                    judgement = judgement_match.group(1).lower()
                    if judgement == "invalid":
                        is_invalid = True
                    elif judgement == "unsure":
                        num_unsure += 1
            elif "unspecified" in rest.lower():
                # If connection is unspecified, mark the entire chain as invalid
                is_invalid = True

            # Store for summary
            all_lines.append(line)

    if not words_set:
        return None, None, None, None

    # Create summary of all judgement lines
    summary = "\n".join(all_lines)

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

    # If there's an invalid connection, score is 0
    if is_invalid:
        return 0.0, "Chain contains invalid connection"

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


def _get_score(query, response, judge_model, start_word: str, end_word: str):
    """Get score for a wordchain response using a judge model."""
    # There must be an ANSWER: line
    answer_line = _get_line_starting_with(response, "ANSWER:")
    if answer_line is None:
        return 0.0, "No valid 'ANSWER:' line found in response."

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
    if only_answer:
        raise NotImplementedError(
            "only_answer mode is not supported with the new prompt format"
        )

    if not hasattr(prediction, "response") or not prediction.response:
        return dspy.Prediction(score=False, feedback="No response found.")

    score, feedback = _get_score(
        example.query,
        prediction.response,
        judge_model,
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
