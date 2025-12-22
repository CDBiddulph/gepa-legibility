"""Utilities for GEPA experiment configuration and path parsing."""

import json
import os


def get_environment_from_path(path):
    """Extract environment type from experiment path.

    Args:
        path: Path to experiment directory

    Returns:
        Environment name: 'psychosis', 'mcq', or 'wordchain'
    """
    path = str(path)
    if "psychosis" in path:
        if "gameable=never" in path:
            raise NotImplementedError(
                'Psychosis "split" environment is no longer supported'
            )
        return "psychosis"
    elif "mcq" in path:
        return "mcq"
    elif "wordchain" in path:
        return "wordchain"
    else:
        raise ValueError(f"Unknown setting for path: {path}")


def get_executor_model_name_from_path(path):
    """Extract executor model name from config.json in the experiment directory.

    Args:
        path: Path to experiment directory (e.g., logs/psychosis/.../0/)

    Returns:
        Model name in format 'provider/model-name'
    """
    # Find config.json in this path or subdirectories
    # It will probably end up finding <path>/0/config.json
    for root, dirs, files in os.walk(path):
        if "config.json" in files:
            config_path = os.path.join(root, "config.json")
            with open(config_path) as f:
                config = json.load(f)
            if "executor_name" in config:
                return config["executor_name"]

    raise ValueError(f"No config.json with executor_name found in {path}")


def uses_incompetent_adapter(experiment_path):
    """Check if an experiment path indicates IncompetentAdapter should be used.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Boolean indicating whether IncompetentAdapter should be used
    """
    return any(x in str(experiment_path).lower() for x in ["-incompetent", "-forget"])
