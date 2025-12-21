"""Path generation utilities for GEPA experiments.

Generates directory paths based on swept parameters using a uniform format:
  field_name=sanitized_value-other_field=other_value

All values are sanitized to be valid directory names:
  - `/` → `_` (for model names like anthropic/claude-sonnet)
  - `-` → `_` (so `-` only appears as field separator)
"""

import dataclasses
import os

# Fields that are metadata/runtime, not experiment configuration.
# These are excluded from path generation.
_EXCLUDED_PATH_FIELDS = {
    "env_name",
    "experiment_name",
    "date_str",
    "cache",
    "seed",
    "log_dir_index",
    "sweep_fields",
}


def _sanitize_value(value) -> str:
    """Sanitize a value for use in a directory name."""
    s = str(value)
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    return s


def _get_config_fields(config) -> list[str]:
    """Get all config fields that should appear in paths (excludes metadata)."""
    return [f.name for f in dataclasses.fields(config) if f.name not in _EXCLUDED_PATH_FIELDS]


def _format_swept_path(sweep_fields: list[str], config) -> str:
    """Generate path component based on parameters."""
    fields_to_use = sweep_fields if sweep_fields else _get_config_fields(config)

    if not fields_to_use:
        return "default"

    parts = []
    for field in fields_to_use:
        value = getattr(config, field)
        sanitized = _sanitize_value(value)
        parts.append(f"{field}={sanitized}")

    return "-".join(parts)


def make_log_dir(config) -> str:
    """Create the log directory path for a trial.

    Args:
        config: Config dataclass with env_name, experiment_name, date_str,
                sweep_fields, and log_dir_index fields.

    Returns:
        The created log directory path.
    """
    task = config.env_name

    if config.experiment_name:
        base_path = f"logs/{task}/{config.experiment_name}/{config.date_str}/"
    else:
        base_path = f"logs/{task}/{config.date_str}/"

    varied_path = _format_swept_path(config.sweep_fields or [], config)

    log_dir = f"{base_path}{varied_path}/"
    if config.log_dir_index is not None:
        log_dir += f"{config.log_dir_index}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
