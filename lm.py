import functools
import litellm
import json


def get_lm_kwargs(model):
    if model == "openai/custom":
        return {"api_base": "http://localhost:8000/v1"}
    if any(model.endswith(s) for s in ["R1", "V3.1", "V3.1-Terminus", "o4-mini", "o3"]):
        return {"reasoning_effort": "high"}
    return {}


def get_litellm_lm(model, max_tokens=32000, temperature=1.0, cache=True):
    maxsize = None if cache else 0

    # O-series models only support temperature 1.0
    if any(model.endswith(s) for s in ["o4-mini", "o3"]) and temperature != 1.0:
        print(f"Changing temperature from {temperature} to 1.0 for {model}")
        temperature = 1.0

    # Use json to make the messages hashable
    @functools.lru_cache(maxsize=maxsize)
    def _call(messages_json):
        messages = json.loads(messages_json)
        return (
            litellm.completion(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **get_lm_kwargs(model),
            )
            .choices[0]
            .message.content.strip()
        )

    return lambda messages: _call(json.dumps(messages))
