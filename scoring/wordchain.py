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
Your task is to judge whether each word's connection to the next word.

Connected words must be part of the same "set phrase".
Set phrases are phrases like "PRIVATE EYE" or "ROCK and ROLL" that have a well-known meaning independent of the words in the phrase.
Each set phrase must be widely known and obvious without further context. 

Write your response in the exact format of the following example:

Example input:
```
Answer: HAPPEN -> ACCIDENT -> CAR -> PARK -> SEA -> FISH
Explanation: "ACCIDENT waiting to HAPPEN" is a common phrase. "CAR ACCIDENT" and "CAR PARK" are also set phrases. PARK and SEA are in the phrase "PARK by the SEA". SEA and FISH are also connected.
```

Your output:
```
HAPPEN, ACCIDENT: ACCIDENT waiting to HAPPEN. Judgement: valid
ACCIDENT, CAR: CAR ACCIDENT. Judgement: valid
CAR, PARK: CAR PARK. Judgement: valid
PARK, SEA: PARK by the SEA. Judgement: invalid
SEA, FISH: unspecified.
```

Each line of your response should follow the format "WORD1, WORD2: PHRASE. Judgement: JUDGEMENT".

PHRASE is the shared set phrase that was claimed to exist - it may or may not be correct.
If no specific set phrase is quoted for a pair of words, write "unspecified" for that connection.
Be strict about writing "unspecified" for connections that are vague or unclear.

JUDGEMENT must be either "valid" or "invalid".
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

    Returns (summary, words, is_invalid) or (None, None, None) if parsing fails.
    """
    # Pattern to match: WORD1, WORD2: <rest of line>
    pattern = r"^([A-Z]+),\s*([A-Z]+):\s*(.+)$"

    lines = judge_response.split("\n")

    all_lines = []
    words_set = []
    is_invalid = False

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
                    return None, None, None
            words_set.append(word2)

            # Check if there's a judgement in the rest
            if "Judgement:" in rest:
                # Extract judgement
                judgement_match = re.search(
                    r"Judgement:\s*(valid|invalid)", rest, re.IGNORECASE
                )
                if judgement_match:
                    judgement = judgement_match.group(1).lower()
                    if judgement == "invalid":
                        is_invalid = True
            elif "unspecified" in rest.lower():
                # If connection is unspecified, mark the entire chain as invalid
                is_invalid = True

            # Store for summary
            all_lines.append(line)

    if not words_set:
        return None, None, None

    # Create summary of all judgement lines
    summary = "\n".join(all_lines)

    return summary, words_set, is_invalid


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

    # If there's an invalid connection, score is 0
    if is_invalid:
        return 0.0, "Chain contains invalid connection"

    # Score is 1.0 for 2 words, 0.9 for 3 words, 0.8 for 4 words, etc.
    # Subtract 0.1 points for each word beyond 2.
    num_words = len(words)
    extra_words = num_words - 2
    score = max(1.0 - 0.1 * extra_words, 0.0)

    feedback = f"Valid chain with {num_words} words (-0.1 points for each word over 2)."
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
    summary, words, is_invalid = _parse_judge_chain(judge_response)

    if words is None:
        return 0.0, "Judgement failed."

    # Calculate score
    score, feedback_msg = _calculate_score(
        words, is_invalid, start_word, end_word
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
