"""JSON parsing utilities."""

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
        json_str = f"{json_str[:1000]}..." if len(json_str) > 1000 else json_str
        logging.error(f"Error parsing JSON: {json_str}")
        return None

def get_json_response(model: str, prompt: str) -> dict:
    """Get a JSON response from the LLM, retrying (without caching) if it fails."""
    text_response = get_simple_lm(model)(prompt)
    json_response = None
    for _ in range(5):
        json_response = extract_json_from_response(text_response)
        if json_response is not None:
            return json_response
        text_response = get_simple_lm(model, cache=False)(prompt)
    raise ValueError(f"Failed to extract JSON from response: {text_response}")

