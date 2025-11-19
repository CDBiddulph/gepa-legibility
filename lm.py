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
        result["api_base"] = "http://localhost:8042/v1"
    if any(model.endswith(s) for s in REASONING_MODELS):
        result["reasoning_effort"] = "high"
        result["allowed_openai_params"] = ["reasoning_effort"]
    return result


def get_dspy_lm(model, max_tokens=32000, temperature=1.0, cache=True, **kwargs):
    """Create a DSPy LM instance.

    Supports:
    - Standard models: "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022", etc.
    - Local models: "local/allenai/OLMo-2-1124-13B-SFT" (via filesystem queue worker)
    """
    # For local/ models, use openai/custom as the model name
    # For now, the original `model` name is unused, but we will later validate that the currently running worker is serving the correct model.
    dspy_model = model
    if model.startswith("local/"):
        dspy_model = "openai/custom"

    # Get model-specific kwargs
    model_kwargs = get_lm_kwargs(dspy_model)

    # Merge all kwargs
    all_kwargs = {**model_kwargs, **kwargs}

    if not any(dspy_model.endswith(s) for s in REASONING_MODELS):
        # Limit the number of tokens for non-reasoning models to avoid token limit errors
        max_tokens = min(max_tokens, 16000)

    # Create LM
    lm = dspy.LM(
        dspy_model,
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
