import dspy


def get_lm_kwargs(model):
    if model == "openai/custom":
        return {"api_base": "http://localhost:8000/v1"}
    if any(model.endswith(s) for s in ["R1", "V3.1", "V3.1-Terminus", "o4-mini", "o3"]):
        return {"reasoning_effort": "high"}
    return {}


def get_dspy_lm(
    model, max_tokens=32000, temperature=1.0, cache=True, history=None, **kwargs
):
    """
    DSPy-based language model interface with automatic caching and reasoning support.
    
    This is a drop-in replacement for the previous litellm-based implementation,
    providing superior caching (disk + memory) and automatic parameter fixing.
    """
    
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
            **all_kwargs
        )
    except ValueError as e:
        if "require passing temperature=1.0" in str(e):
            print(f"Changing temperature from {temperature} to 1.0 for {model}")
            lm = dspy.LM(
                model, 
                temperature=1.0, 
                max_tokens=max_tokens, 
                cache=cache,
                **all_kwargs
            )
        else:
            raise
    
    # Add history support if requested
    if history is not None:
        original_call = lm.__call__
        
        def call_with_history(*args, **call_kwargs):
            result = original_call(*args, **call_kwargs)
            
            # DSPy should always populate history after a call
            raw_response = lm.history[-1]['response']
            messages = lm.history[-1]['messages']
            message = raw_response.choices[0].message
            full_response = message.content
            
            # Add reasoning content if available (for models like deepseek-reasoner)
            if message.reasoning_content:
                full_response = f"<thinking>{message.reasoning_content}</thinking>{full_response}"
            
            history.append((messages, full_response))
            return result
        
        lm.__call__ = call_with_history
    
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
        assert len(result) == 1, f"Expected exactly 1 result from DSPy LM, got {len(result)}"
        return result[0]
    
    return lm_callable