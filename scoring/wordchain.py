import functools
import re
import dspy
import sys
import os

# Add parent directory to path to import lm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm import get_simple_lm


DEFAULT_JUDGE_MODEL_NAME = "openai/gpt-4.1-mini"

# TODO: add a prompt for getting the sentiment of the response for each connection


def _get_judge_model(model=DEFAULT_JUDGE_MODEL_NAME):
    # If it's already a callable model object, return it directly
    if callable(model) and not isinstance(model, str):
        return model
    # Otherwise, assume it's a model name string and create the model
    return get_simple_lm(model, temperature=0)


DEFAULT_JUDGE_MODEL = _get_judge_model(DEFAULT_JUDGE_MODEL_NAME)

JUDGE_PROMPT = """
See the above text content, which should contain a sequence of words and set phrases that connect them.
Your task is to judge whether and how each word is connected to the next word.

Each pair of adjacent words must be part of the same set phrase.
Set phrases are phrases like "fast food," "beat around the bush," or "long time no see" whose meanings cannot be fully derived from their individual words.
The adjacent words may appear in the set phrase in either order.

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

JUDGEMENT must be either "valid" or "invalid".
No judgement is needed for "unspecified" connections.
Don't be too harsh or too lenient in your judgements.

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
        The judgement ('valid' or 'invalid')

    Raises:
        JudgeParseError: If judgement format is incorrect
    """
    judgement_match = re.search(
        r"\(judgement:\s*(valid|invalid)\)", rest, re.IGNORECASE
    )

    if not judgement_match:
        raise JudgeParseError(
            f"Line {line_num}: Judgement not in format (judgement: valid/invalid)"
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
        Tuple of (word1, word2, phrase, is_invalid_line)

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
        # Use a placeholder phrase for unspecified connections
        return word1, word2, "UNSPECIFIED", True

    # For non-unspecified lines, parse the quoted phrase and judgement
    phrase = _extract_and_validate_quoted_phrase(rest, word1, word2, line_num)
    judgement = _extract_judgement(rest, line_num)

    is_invalid_line = judgement == "invalid"

    return word1, word2, phrase, is_invalid_line


def _parse_judge_chain(judge_response):
    """
    Parse the judge's response to extract the word chain.
    Expected format:
    - "WORD1, WORD2: "PHRASE" (judgement: JUDGEMENT)"
    - "WORD1, WORD2: unspecified"

    Returns:
        Tuple of (words, phrases, invalid_connections)

    Raises:
        JudgeParseError: If the response cannot be parsed
    """
    lines = judge_response.split("\n")

    words = []
    phrases = []
    invalid_connections = []

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        result = _parse_judge_line(line, line_num, words)
        if result is None:
            continue  # Skip non-word-pair lines

        word1, word2, phrase, is_invalid_line = result

        # Add to word chain
        if len(words) == 0:
            words.append(word1)
        words.append(word2)
        phrases.append(phrase)

        # Track invalid connections
        if is_invalid_line:
            invalid_connections.append((word1, word2))

    if not words:
        raise JudgeParseError("No valid word pairs found in judge response")

    return words, phrases, invalid_connections

def _calculate_score(words, invalid_connections, start_word, end_word):
    """
    Calculate score based on the word chain.
    Returns (score, feedback).
    """
    # Verify first and last words match
    if not words or len(words) < 2:
        return 0.0, "Chain must have at least 2 words"

    if words[0] != start_word:
        return 0.0, f"First word '{words[0]}' is not '{start_word}'"

    if words[-1] != end_word:
        return 0.0, f"Last word '{words[-1]}' is not '{end_word}'"

    # If there's an invalid connection, score is 0
    if invalid_connections:
        invalid_connections_str = ", ".join(f'"{w1} → {w2}"' for w1, w2 in invalid_connections)
        return 0.0, f"Chain contains invalid connection(s): {invalid_connections_str}"

    # Score is 1.0 for 2 words, 0.8 for 3 words, 0.6 for 4 words, etc.
    # Subtract 0.1 points for each word beyond 2.
    num_words = len(words)
    extra_words = num_words - 2
    score = max(1.0 - 0.1 * extra_words, 0.0)

    feedback = f"Valid chain with {num_words} words"
    return score, feedback


def _get_judge_response(response, judge_model):
    # Strip any thinking tags from the response before evaluation
    # TODO: make a better, more general solution for this
    response_for_judge = _strip_thinking_tags(response)

    # There must be an ANSWER: line
    answer_line = _get_line_starting_with(response_for_judge, "ANSWER:")
    if answer_line is None:
        raise JudgeParseError("No valid 'ANSWER:' line found in response.")

    # Ask for judgement of the chain
    judge_full_prompt = (
        f"<content>{response_for_judge}</content>\n{JUDGE_PROMPT.strip()}"
    )
    judge_response = judge_model([{"role": "user", "content": judge_full_prompt}])

    # Parse the judge's response
    return _parse_judge_chain(judge_response)


def _get_normalized_response(words, phrases):
    """Normalize the response to a standard format."""
    chain_str = " -> ".join(words)
    explanations = []
    for word1, word2, phrase in zip(words[:-1], words[1:], phrases, strict=True):
        explanations.append(f'{word1} and {word2} appear in the phrase "{phrase}".')
    return f"ANSWER: {chain_str}\n\n{' '.join(explanations)}"


def _get_score(
    response,
    judge_model,
    start_word: str,
    end_word: str,
    normalize_response: bool,
):
    """Get score for a wordchain response using a judge model."""
    try:
        words, phrases, invalid_connections = _get_judge_response(response, judge_model)
    except JudgeParseError as e:
        return 0.0, f"Judgement failed: {e}"

    if normalize_response:
        normalized_response = _get_normalized_response(words, phrases)
        # Don't catch the exception here, because an error should never occur
        words, phrases, invalid_connections = _get_judge_response(
            normalized_response, judge_model
        )

    # Calculate score
    score, feedback = _calculate_score(words, invalid_connections, start_word, end_word)

    return round(score, 2), feedback


# Use caching to avoid an error where the score is different for the same example
@functools.lru_cache(maxsize=None)
def _metric_fn_impl(example, prediction, judge_model, normalize_response: bool):
    """Metric function that provides detailed feedback for GEPA"""
    if not hasattr(prediction, "response") or not prediction.response:
        return dspy.Prediction(score=False, feedback="No response found.")

    score, feedback = _get_score(
        prediction.response,
        judge_model,
        example.start_word,
        example.end_word,
        normalize_response,
    )
    feedback = f"{feedback}\n\nScore: {score}"

    return dspy.Prediction(score=score, feedback=feedback)


# This wrapper has the signature required by GEPA
def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return _metric_fn_impl(
        example, prediction, DEFAULT_JUDGE_MODEL, normalize_response=False
    )


def get_metric_fn(judge_model=DEFAULT_JUDGE_MODEL_NAME, normalize_response=False):
    lm = _get_judge_model(judge_model)
    return lambda example, prediction, trace=None, pred_name=None, pred_trace=None: _metric_fn_impl(
        example, prediction, lm, normalize_response
    )
