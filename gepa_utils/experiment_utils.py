"""Utilities for GEPA experiment configuration and path parsing."""


def get_environment_from_path(path):
    """Extract environment type from experiment path.

    Args:
        path: Path to experiment directory

    Returns:
        Environment name: 'psychosis_single', 'psychosis_split', 'mcq', 'alliterate', or 'wordchain'
    """
    path = str(path)
    if "psychosis" in path:
        if "gameable=never" in path:
            return "psychosis_single"
        elif "gameable=sometimes" in path or "gameable=" not in path:
            return "psychosis_split"
        else:
            raise ValueError(f"Unknown psychosis setting for path: {path}")
    elif "mcq" in path:
        return "mcq"
    elif "alliterate" in path:
        return "alliterate"
    elif "wordchain" in path:
        return "wordchain"
    else:
        raise ValueError(f"Unknown setting for path: {path}")


def get_executor_model_name_from_path(path):
    """Extract executor model name from experiment path.

    Args:
        path: Path to experiment directory

    Returns:
        Model name in format 'openai/model-name'
    """
    executor_substr = str(path).split("-e=")[-1]
    if executor_substr.startswith("gpt-4.1-mini"):
        return "openai/gpt-4.1-mini"
    elif executor_substr.startswith("o4-mini"):
        return "openai/o4-mini"
    elif executor_substr.startswith("gpt-5-mini"):
        return "openai/gpt-5-mini"
    elif executor_substr.startswith("Qwen3-14B"):
        return "deepinfra/Qwen/Qwen3-14B"
    elif executor_substr.startswith("qwen-2.5-7b-instruct"):
        return "openrouter/qwen/qwen-2.5-7b-instruct"
    elif executor_substr.startswith("Qwen3-30B-A3B"):
        return "deepinfra/Qwen/Qwen3-30B-A3B"
    else:
        raise ValueError(f"Unknown executor: {executor_substr}")


def uses_incompetent_adapter(experiment_path):
    """Check if an experiment path indicates IncompetentAdapter should be used.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Boolean indicating whether IncompetentAdapter should be used
    """
    return any(x in str(experiment_path).lower() for x in ["-incompetent", "-forget"])
