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
]


def get_lm_kwargs(model):
    result = {}
    if model == "openai/custom":
        result["api_base"] = "http://localhost:8000/v1"
    if any(model.endswith(s) for s in REASONING_MODELS):
        result["reasoning_effort"] = "high"
        result["allowed_openai_params"] = ["reasoning_effort"]
    return result


def get_dspy_lm(model, max_tokens=32000, temperature=1.0, cache=True, **kwargs):
    """Create a DSPy LM instance."""
    # Get model-specific kwargs
    model_kwargs = get_lm_kwargs(model)

    # Merge all kwargs
    all_kwargs = {**model_kwargs, **kwargs}

    # Create LM
    lm = dspy.LM(
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=cache,
        **all_kwargs,
    )

    return lm


def get_simple_lm(model, **kwargs):
    """
    Get a simple callable that takes messages and returns a string response.
    This is useful for non-DSPy code that just wants a function interface.

    Returns a function: (messages) -> str
    """
    lm = get_dspy_lm(model, **kwargs)

    def lm_callable(messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        result = lm(messages=messages)
        assert (
            len(result) == 1
        ), f"Expected exactly 1 result from DSPy LM, got {len(result)}"
        return result[0]

    return lm_callable
