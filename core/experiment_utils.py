"""Utilities for GEPA experiment configuration and path parsing."""

import json
import os
from pathlib import Path
from typing import Optional

import yaml

# Path to the experiments config file (relative to repo root)
EXPERIMENTS_CONFIG_PATH = Path(__file__).parent.parent / "experiments_config.yaml"


def load_experiments_config(version: Optional[str] = None) -> dict:
    """Load experiment paths from the YAML config file.

    Args:
        version: Config version to load (e.g., "v1", "v2").
                 If None, uses the default_version from config.

    Returns:
        Dict with structure: {env_name: {experiment_type: path}}
        e.g., {"psychosis": {"teacher_true": "logs/psychosis/..."}}

    Example:
        >>> config = load_experiments_config()        # uses default
        >>> config = load_experiments_config("v1")    # uses v1
        >>> config["mcq"]["baseline_only"]
        'logs/mcq/baseline-only/2026-01-06-08-59-06'
    """
    with open(EXPERIMENTS_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    if version is None:
        version = raw["default_version"]

    if version not in raw["versions"]:
        available = list(raw["versions"].keys())
        raise ValueError(f"Unknown version '{version}'. Available: {available}")

    # Return just the environment paths (exclude 'description')
    version_data = raw["versions"][version]
    return {k: v for k, v in version_data.items() if k != "description"}


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


def expand_experiment_paths(path):
    """Expand a path to a list of experiment directories.

    Checks for config.json in run directories:
    - path/*/config.json exists → experiment dir, return [path]
    - path/*/*/config.json exists → parent dir, return all subdirs

    Args:
        path: Path to experiment directory or parent directory (str or Path)

    Returns:
        List of Path objects to experiment directories

    Raises:
        ValueError: If path is not a directory or no valid experiments found
    """
    path = Path(path)

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Case 1: path/*/config.json → this is an experiment dir
    if list(path.glob("*/config.json")):
        return [path]

    # Case 2: path/*/*/config.json → parent of experiment dirs
    subdirs = sorted(d for d in path.iterdir() if d.is_dir())
    experiments = [d for d in subdirs if list(d.glob("*/config.json"))]

    if not experiments:
        raise ValueError(f"No config.json found in {path}/*/ or {path}/*/*/")
    if len(experiments) != len(subdirs):
        invalid = sorted(set(subdirs) - set(experiments))
        raise ValueError(f"Invalid subdirs (no */config.json): {[d.name for d in invalid]}")

    return experiments
