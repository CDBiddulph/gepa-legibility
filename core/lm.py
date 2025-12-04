import dspy
import litellm

# Allow litellm to automatically handle unsupported parameters
# This is necessary for models like GPT-5 that only support temperature=1
litellm.drop_params = True

REASONING_MODELS = [
    "R1",
    "V3.1",
    "V3.1-Terminus",
    "V3.2",
    "V3.2-Exp",
    "o4-mini",
    "o3",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "Qwen3-30B-A3B",
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
    if any(model.endswith(s) for s in REASONING_MODELS):
        result["reasoning_effort"] = "high"
        result["allowed_openai_params"] = ["reasoning_effort"]

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

    # Limit max_tokens for non-reasoning models
    if not any(model.endswith(s) for s in REASONING_MODELS):
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
        return result[0]

    return lm_callable
