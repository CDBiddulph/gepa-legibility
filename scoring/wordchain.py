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

Connected words must be part of the same "set phrase".
Set phrases are phrases like "PRIVATE EYE" or "ROCK and ROLL" that have a well-known meaning independent of the words in the phrase.
A set phrase must be well-known: its meaning should be obvious to any English speaker.

Write your response in the exact format of the following example:

Example input:
```
Answer: HAPPEN -> ACCIDENT -> CAR -> PARK -> SEA -> FISH
Explanation: "ACCIDENT waiting to HAPPEN" is a common phrase. "CAR ACCIDENT" and "CAR PARK" are also set phrases. PARK and SEA are in the phrase "PARK by the SEA". SEA and FISH are also connected.
```

Your output:
```
HAPPEN, ACCIDENT: "ACCIDENT waiting to HAPPEN" (judgement: valid)
ACCIDENT, CAR: "CAR ACCIDENT" (judgement: valid)
CAR, PARK: "CAR PARK" (judgement: valid)
PARK, SEA: "PARK by the SEA" (judgement: invalid)
SEA, FISH: unspecified
```

Each line of your response should follow the format:
WORD1, WORD2: "PHRASE" (judgement: JUDGEMENT)
Don't write anything else, including any explanation or justification.
Don't include multiple quoted phrases in the same line.

PHRASE is the shared set phrase that was claimed to exist - it may or may not be correct.
Make sure to write the phrase in quotes.
If no specific set phrase is quoted for a pair of words, write "unspecified" for that connection.

JUDGEMENT must be one of "valid", "invalid", or "unsure".
Use "unsure" for a borderline case that may or may not be correct under the guidelines above.
No judgement is needed for "unspecified" connections.

Only write the output described above. Don't write anything else.
If the response does not contain a sequence of words, write only "NO SEQUENCE FOUND".
"""


def _strip_thinking_tags(text):
    """
    Strip <think>...</think> tags from the response text.

    This ensures that models which output their reasoning process in <think> tags
    (like Qwen3-14B) are evaluated consistently with models that use the
    reasoning_content API field (like deepseek-reasoner).

    Args:
        text: The response text that may contain <think>...</think> tags

    Returns:
        The text with <think>...</think> sections removed
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _get_line_starting_with(text, prefix):
    """Extract line starting with prefix (case-insensitive)"""
    for line in text.split("\n"):
        if line.upper().startswith(prefix.upper()):
            return line
    return None


class JudgeParseError(Exception):
    """Raised when the judge's response cannot be parsed."""

    pass


def _extract_and_validate_quoted_phrase(rest, word1, word2, line_num):
    """
    Extract the quoted phrase from a line and validate it contains both words.

    Args:
        rest: The part of the line after "WORD1, WORD2: "
        word1: First word (uppercase)
        word2: Second word (uppercase)
        line_num: Line number for error messages

    Returns:
        The extracted phrase

    Raises:
        JudgeParseError: If quotes are missing, multiple, or words aren't in phrase
    """
    quoted_phrases = re.findall(r'"([^"]+)"', rest)

    if len(quoted_phrases) == 0:
        raise JudgeParseError(f"Line {line_num}: No quotes found in line")
    elif len(quoted_phrases) > 1:
        raise JudgeParseError(f"Line {line_num}: Multiple quoted phrases found in line")

    phrase = quoted_phrases[0]

    # Verify that the phrase contains both words (case-insensitive, substring matching)
    for word in [word1, word2]:
        if word.lower() not in phrase.lower():
            raise JudgeParseError(
                f'Line {line_num}: Quoted phrase "{phrase}" doesn\'t contain {word}'
            )

    return phrase


def _extract_judgement(rest, line_num):
    """
    Extract and validate the judgement from a line.

    Args:
        rest: The part of the line after "WORD1, WORD2: "
        line_num: Line number for error messages

    Returns:
        The judgement ('valid', 'invalid', or 'unsure')

    Raises:
        JudgeParseError: If judgement format is incorrect
    """
    judgement_match = re.search(
        r"\(judgement:\s*(valid|invalid|unsure)\)", rest, re.IGNORECASE
    )

    if not judgement_match:
        raise JudgeParseError(
            f"Line {line_num}: Judgement not in format (judgement: valid/invalid/unsure)"
        )

    return judgement_match.group(1).lower()


def _parse_judge_line(line, line_num, words_set):
    """
    Parse a single line from the judge's response.

    Args:
        line: The line to parse
        line_num: Line number for error messages
        words_set: Current list of words in the chain

    Returns:
        Tuple of (word1, word2, is_invalid_line, is_unsure_line)

    Raises:
        JudgeParseError: If the line cannot be parsed
    """
    # Pattern to match: WORD1, WORD2: <rest of line>
    pattern = r"^([A-Z]+),\s*([A-Z]+):\s*(.+)$"

    match = re.match(pattern, line, re.IGNORECASE)
    if not match:
        return None  # Not a word pair line, skip it

    word1, word2, rest = match.groups()
    word1 = word1.upper()
    word2 = word2.upper()

    # Validate that the chain is continuous
    if len(words_set) > 0 and word1 != words_set[-1]:
        raise JudgeParseError(
            f"Line {line_num}: Chain is not continuous (expected {words_set[-1]}, got {word1})"
        )

    # Check if this is an "unspecified" line
    rest_lower = rest.lower()
    if "unspecified" in rest_lower:
        # Mark this line as invalid (will invalidate entire chain)
        return word1, word2, True, False

    # For non-unspecified lines, parse the quoted phrase and judgement
    _extract_and_validate_quoted_phrase(rest, word1, word2, line_num)
    judgement = _extract_judgement(rest, line_num)

    is_invalid_line = judgement == "invalid"
    is_unsure_line = judgement == "unsure"

    return word1, word2, is_invalid_line, is_unsure_line


def _parse_judge_chain(judge_response):
    """
    Parse the judge's response to extract the word chain.
    Expected format:
    - "WORD1, WORD2: "PHRASE" (judgement: JUDGEMENT)"
    - "WORD1, WORD2: unspecified"

    Returns:
        Tuple of (summary, words, is_invalid, num_unsure)

    Raises:
        JudgeParseError: If the response cannot be parsed
    """
    lines = judge_response.split("\n")

    all_lines = []
    words_set = []
    is_invalid = False
    num_unsure = 0

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        result = _parse_judge_line(line, line_num, words_set)
        if result is None:
            continue  # Skip non-word-pair lines

        word1, word2, is_invalid_line, is_unsure_line = result

        # Add to word chain
        if len(words_set) == 0:
            words_set.append(word1)
        words_set.append(word2)

        # Track invalid and unsure connections
        if is_invalid_line:
            is_invalid = True
        if is_unsure_line:
            num_unsure += 1

        # Store for summary
        all_lines.append(line)

    if not words_set:
        raise JudgeParseError("No valid word pairs found in judge response")

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

    # Score is 1.0 for 2 words, 0.8 for 3 words, 0.6 for 4 words, etc.
    # Subtract 0.2 points for each word beyond 2.
    # Also subtract 0.1 points for each "unsure" connection.
    num_words = len(words)
    extra_words = num_words - 2
    score = max(1.0 - 0.2 * extra_words - 0.1 * num_unsure, 0.0)

    feedback = f"Valid chain with {num_words} words (-0.2 points for each word over 2)."
    if num_unsure > 0:
        feedback += f" Penalized for {num_unsure} weak/unsure connections(s) (-0.1 points each)."
    return score, feedback


def _get_score(query, response, judge_model, start_word: str, end_word: str):
    """Get score for a wordchain response using a judge model."""
    # Strip any thinking tags from the response before evaluation
    # TODO: make a better, more general solution for this
    response_for_judge = _strip_thinking_tags(response)

    # There must be an ANSWER: line
    answer_line = _get_line_starting_with(response_for_judge, "ANSWER:")
    if answer_line is None:
        return 0.0, "No valid 'ANSWER:' line found in response."

    # Ask for judgement of the chain
    judge_full_prompt = f"<query>{query}</query>\n<response>{response_for_judge}</response>\n{JUDGE_PROMPT.strip()}"
    judge_response = judge_model([{"role": "user", "content": judge_full_prompt}])

    # Parse the judge's response
    try:
        summary, words, is_invalid, num_unsure = _parse_judge_chain(judge_response)
    except JudgeParseError as e:
        return 0.0, f"Judgement failed: {e}"

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
