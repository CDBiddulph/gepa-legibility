import functools
import litellm
import json


def get_lm_kwargs(model):
    if model == "openai/custom":
        return {"api_base": "http://localhost:8000/v1"}
    if any(model.endswith(s) for s in ["R1", "V3.1", "V3.1-Terminus", "o4-mini", "o3"]):
        return {"reasoning_effort": "high"}
    return {}


def _call(kwargs_json):
    kwargs = json.loads(kwargs_json)
    return litellm.completion(**kwargs)


# Use json to make the messages hashable
# TODO: Use better caching mechanism, like in https://docs.litellm.ai/docs/proxy/caching
@functools.lru_cache(maxsize=None)
def _call_cached(kwargs_json):
    return _call(kwargs_json)


def get_litellm_lm(
    model, max_tokens=32000, temperature=1.0, cache=True, judge_history=None
):
    # O-series models only support temperature 1.0
    if any(model.endswith(s) for s in ["o4-mini", "o3"]) and temperature != 1.0:
        print(f"Changing temperature from {temperature} to 1.0 for {model}")
        temperature = 1.0

    kwargs_base = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **get_lm_kwargs(model),
    }

    def _call_outer(messages):
        kwargs_full = kwargs_base.copy()
        kwargs_full["messages"] = messages
        kwargs_json = json.dumps(kwargs_full)
        call_fn = _call_cached if cache else _call
        call_result = call_fn(kwargs_json)
        message = call_result.choices[0].message
        if judge_history is not None:
            full_response = message.content
            if hasattr(message, "reasoning_content"):
                full_response = (
                    f"<thinking>{message.reasoning_content}</thinking>{full_response}"
                )
            judge_history.append((messages, full_response))
        return message.content.strip()

    return _call_outer
