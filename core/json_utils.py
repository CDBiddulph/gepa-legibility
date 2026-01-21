"""JSON parsing utilities."""

from typing import Callable
import json
import logging

from core.lm import get_simple_lm


def extract_json_from_response(content: str) -> dict | list | None:
    """Extract JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response that may contain JSON in a code block

    Returns:
        Parsed JSON object, or None if parsing fails
    """
    if "```json\n" in content:
        start = content.find("```json\n") + 8
        end = content.find("\n```", start)
        json_str = content[start:end].strip() if end != -1 else content[start:].strip()
    elif "```\n" in content:
        start = content.find("```\n") + 4
        end = content.find("\n```", start)
        json_str = content[start:end].strip() if end != -1 else content[start:].strip()
    else:
        json_str = content.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fallback: find first line with only "{" and last line with only "}"
    # This handles cases where the model adds extra content before/after JSON
    result = _extract_json_by_braces(content)
    if result is not None:
        return result

    logging.error(f"Error parsing JSON: {json_str}")
    return None


def _extract_json_by_braces(content: str) -> dict | list | None:
    """Fallback JSON extraction by finding lines with only { and }.

    Finds the first line containing only "{" and the last line containing only "}",
    then attempts to parse everything between (inclusive) as JSON.

    Args:
        content: Raw response text

    Returns:
        Parsed JSON object, or None if extraction/parsing fails
    """
    lines = content.split("\n")

    # Find first line that is just "{"
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "{":
            start_idx = i
            break

    if start_idx is None:
        return None

    # Find last line that is just "}"
    end_idx = None
    for i in range(len(lines) - 1, start_idx, -1):
        if lines[i].strip() == "}":
            end_idx = i
            break

    if end_idx is None:
        return None

    # Extract and parse the JSON
    json_str = "\n".join(lines[start_idx:end_idx + 1])
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def get_json_response(
    model: str, prompt: str, validation_fn: Callable[[dict], bool] = None
) -> dict:
    """Get a JSON response from the LLM, retrying (without caching) if it fails.

    Args:
        model: The model to use
        prompt: The prompt to send to the model
        validation_fn: A function to validate the JSON response, which should return True if the response is valid and False otherwise

    Returns:
        The valid JSON response
    """
    text_response = get_simple_lm(model)(prompt)
    json_response = None
    # By default, use a validation function that always succeeds
    validation_fn = validation_fn or (lambda _: True)
    for _ in range(5):
        json_response = extract_json_from_response(text_response)
        if json_response is not None and validation_fn(json_response):
            return json_response
        text_response = get_simple_lm(model, cache=False)(prompt)
    raise ValueError(f"Failed to extract valid JSON from response: {text_response}")
