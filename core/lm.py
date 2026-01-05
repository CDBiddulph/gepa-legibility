import dspy
import litellm

# Allow litellm to automatically handle unsupported parameters
# This is necessary for models like GPT-5 that only support temperature=1
litellm.drop_params = True

HIGH_REASONING_MODELS = [
    # DeepSeek models
    "R1",
    "V3.1",
    "V3.1-Terminus",
    "V3.2",
    "V3.2-Exp",
    # OpenAI models
    "o4-mini",
    "o3",
    "gpt-5",
    "gpt-5.2",
    # Anthropic models
    "claude-opus-4-5",
    # Google models
    "gemini-3-pro-preview",
    # Qwen models
    "Qwen3-30B-A3B",
]

MINIMAL_REASONING_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano",
]


def get_lm_kwargs(model):
    """Get model-specific kwargs for litellm/DSPy."""
    result = {}

    # Set api_base for custom endpoints
    if model.startswith("local/"):
        result["api_base"] = "http://localhost:8042/v1"
        result["custom_llm_provider"] = "openai"
    elif model.startswith("tinker/"):
        result["api_base"] = "http://localhost:8043/v1"
        result["custom_llm_provider"] = "openai"

    # Set reasoning parameters for reasoning models
    if any(model.endswith(s) for s in HIGH_REASONING_MODELS):
        result["reasoning_effort"] = "high"
    elif any(model.endswith(s) for s in MINIMAL_REASONING_MODELS):
        result["reasoning_effort"] = "minimal"

    return result


def get_dspy_lm(model, max_tokens=32000, temperature=1.0, cache=True, **kwargs):
    """Create a DSPy LM instance.

    Supports:
    - Standard models: "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022", etc.
    - Local models: "local/model-name" (via filesystem queue worker on port 8042)
    - Tinker models: "tinker/model-name" (via tinker_server.py on port 8043)
    """
    # Get model-specific kwargs (api_base, reasoning params, etc.)
    model_kwargs = get_lm_kwargs(model)

    # Merge user kwargs with model-specific kwargs
    all_kwargs = {**model_kwargs, **kwargs}

    # Limit max_tokens for non-reasoning models and non-Tinker models
    if not any(model.endswith(s) for s in HIGH_REASONING_MODELS) and not model.startswith("tinker/"):
        max_tokens = min(max_tokens, 16000)

    return dspy.LM(
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=cache,
        **all_kwargs,
    )


def get_simple_lm(model, **kwargs):
    """Get a simple callable that takes messages and returns a string response.

    This is useful for non-DSPy code that just wants a function interface.

    Args:
        model: Model name (same format as get_dspy_lm)
        **kwargs: Additional arguments passed to get_dspy_lm

    Returns:
        Callable that takes messages (str or list[dict]) and returns str
    """
    lm = get_dspy_lm(model, **kwargs)

    def lm_callable(messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        result = lm(messages=messages)
        assert len(result) == 1, f"Expected 1 result from LM, got {len(result)}"
        output = result[0]
        # Handle dict output from reasoning models (which include reasoning_content)
        if isinstance(output, dict):
            output = output["text"]
        return output

    return lm_callable


def _extract_reasoning_from_output(text: str) -> tuple[str | None, str]:
    """Extract reasoning and response from raw LM output with <think> tags.

    Handles models that output <think>...</think> tags (like Tinker/Qwen models).
    Also handles cases where the opening <think> tag is missing.

    Args:
        text: Raw LM output text

    Returns:
        (reasoning, response) where reasoning is the content inside <think> tags
        (or None if no tags present), and response is the text after </think>.
    """
    think_closer_idx = text.find("</think>")
    if think_closer_idx == -1:
        return None, text

    think_opener_idx = text.find("<think>")
    if think_opener_idx == -1:
        # Missing opening tag (common with Tinker/Qwen models)
        reasoning_start = 0
    else:
        reasoning_start = think_opener_idx + len("<think>")

    reasoning = text[reasoning_start:think_closer_idx].strip()
    response = text[think_closer_idx + len("</think>") :].strip()
    return reasoning, response


def extract_reasoning_from_history_entry(entry: dict) -> tuple[str | None, str]:
    """Extract reasoning and response from a history entry.

    Handles both:
    1. Native reasoning_content field (for models like deepseek-reasoner)
    2. Inline <think>...</think> tags (for Tinker/Qwen models)

    Args:
        entry: A single entry from lm.history

    Returns:
        (reasoning, response) tuple
    """
    # Try to get native reasoning_content from the response object
    try:
        choices = entry["response"]["choices"]
        assert len(choices) == 1, "Expected exactly one choice"
        message = choices[0].message
        reasoning_content = message.get("reasoning_content")
        if reasoning_content and reasoning_content.strip():
            content = message.get("content") or message.content
            return reasoning_content.strip(), content.strip()
    except (KeyError, AttributeError, TypeError):
        pass

    # Fall back to extracting <think> tags from the raw output text
    raw_output = entry["outputs"][0]
    if isinstance(raw_output, dict):
        raw_output = raw_output.get("text")
    if raw_output is None:
        return None, ""
    return _extract_reasoning_from_output(raw_output)


# =============================================================================
# History matching utilities
# =============================================================================


def get_history_content(entry: dict) -> str:
    """Extract message content from a history entry for matching.

    Args:
        entry: A single entry from lm.history

    Returns:
        Concatenated content of all messages in the entry
    """
    messages = entry.get("messages", [])
    return " ".join(msg.get("content", "") for msg in messages)


def find_history_entry(
    history: list[dict],
    input_values: list[str],
) -> dict:
    """Find a history entry containing all input values.

    Used to match evaluation results back to their corresponding LM history
    entries when running parallel evaluation.

    Args:
        history: List of history entries to search.
        input_values: Values that must all appear in the entry's message content

    Returns:
        The matching history entry

    Raises:
        ValueError: If no match is found
    """
    matches = []
    for i, entry in enumerate(history):
        content = get_history_content(entry)
        if all(str(value) in content for value in input_values):
            matches.append(i)

    if len(matches) == 0:
        raise ValueError(
            f"No match found in history ({len(history)} entries) "
            f"for input values: {input_values}"
        )

    # If multiple matches (e.g., from retries), use the last one
    return history[matches[-1]]
