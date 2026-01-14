"""Low-level judge calling utilities."""

from core.json_utils import extract_json_from_response
from core.lm import get_simple_lm


_JUDGE_MODEL = "openai/gpt-5"


def call_judge(
    prompt: str,
    valid_results: tuple[str, ...],
    cache: bool = True,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Call the judge model and parse the response, with retry on failure.

    Args:
        prompt: Complete prompt to send to the judge
        valid_results: Tuple of valid result values (e.g., ("yes", "no", "unclear"))
        cache: Whether to use LM caching. Default True for first attempt.
        max_retries: Maximum number of retry attempts after initial failure.

    Returns:
        Tuple of (result, reasoning) where result is one of valid_results

    Raises:
        ValueError: If response cannot be parsed after all retries
    """
    last_error = None

    for attempt in range(1 + max_retries):
        # First attempt uses cache setting, retries always disable cache
        use_cache = cache if attempt == 0 else False

        try:
            judge_lm = get_simple_lm(_JUDGE_MODEL, temperature=0, cache=use_cache)
            response = judge_lm([{"role": "user", "content": prompt}])

            parsed = extract_json_from_response(response)
            if parsed is None:
                raise ValueError(f"Failed to parse judge response: {response}")

            result = parsed["result"].lower()
            reasoning = parsed.get("reasoning", "")

            if result not in valid_results:
                raise ValueError(f"Invalid result '{result}', expected one of {valid_results}")

            return result, reasoning

        except (ValueError, KeyError) as e:
            last_error = e
            if attempt < max_retries:
                print(f"Judge call failed (attempt {attempt + 1}/{1 + max_retries}), retrying: {e}")

    raise ValueError(f"Judge call failed after {1 + max_retries} attempts. Last error: {last_error}")
