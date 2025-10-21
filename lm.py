import dspy
import json
import os


def get_lm_kwargs(model):
    if model == "openai/custom":
        return {"api_base": "http://localhost:8000/v1"}
    if any(model.endswith(s) for s in ["R1", "V3.1", "V3.1-Terminus", "o4-mini", "o3"]):
        return {"reasoning_effort": "high"}
    return {}


def _setup_history(lm, history_file):
    # Load existing history if it exists
    persistent_history = []
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        with open(history_file, "r") as f:
            persistent_history = json.load(f)

    # Wrap forward to save calls incrementally
    original_forward = lm.forward

    def forward_with_persistent_history(*args, **call_kwargs):
        result = original_forward(*args, **call_kwargs)

        # Extract call info
        prompt = call_kwargs.get("prompt")
        messages = call_kwargs.get("messages")

        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            return result

        # Extract response
        if hasattr(result, "choices") and len(result.choices) > 0:
            message = result.choices[0].message
            response = message.content

            # Add reasoning content if available
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                response = (
                    f"<thinking>\n{message.reasoning_content}\n</thinking>\n{response}"
                )
        else:
            return result

        # Save to persistent history
        persistent_history.append([messages, response])

        # Write incrementally to file
        with open(history_file, "w") as f:
            json.dump(persistent_history, f)

        return result

    lm.forward = forward_with_persistent_history

    # Store reference to persistent history for later access
    lm._persistent_history = persistent_history


def get_dspy_lm(
    model, max_tokens=32000, temperature=1.0, cache=True, history_file=None, **kwargs
):
    # Get model-specific kwargs
    model_kwargs = get_lm_kwargs(model)

    # Merge all kwargs
    all_kwargs = {**model_kwargs, **kwargs}

    # Create LM with auto-fix for reasoning models
    try:
        lm = dspy.LM(
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=cache,
            **all_kwargs,
        )
    except ValueError as e:
        if "require passing temperature=1.0" in str(e):
            print(f"Changing temperature from {temperature} to 1.0 for {model}")
            lm = dspy.LM(
                model, temperature=1.0, max_tokens=max_tokens, cache=cache, **all_kwargs
            )
        else:
            raise

    # Add persistent history tracking if requested
    if history_file is not None:
        _setup_history(lm, history_file)

    return lm


def get_simple_lm(model, **kwargs):
    """
    Get a simple callable that takes messages and returns a string response.
    This is useful for non-DSPy code that just wants a function interface.

    Returns a function: (messages) -> str
    """
    lm = get_dspy_lm(model, **kwargs)

    def lm_callable(messages):
        result = lm(messages=messages)
        assert (
            len(result) == 1
        ), f"Expected exactly 1 result from DSPy LM, got {len(result)}"
        return result[0]

    return lm_callable
