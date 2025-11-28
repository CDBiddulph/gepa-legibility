"""Shared configuration utilities for GEPA optimization."""

import dataclasses
import json
import os


def shorten_model_name(model_name: str) -> str:
    """Extract the short name from a full model path."""
    return model_name.split("/")[-1]


def save_config(log_dir: str, config: dataclasses.dataclass) -> None:
    """Save configuration to config.json in the log directory."""
    config_dict = dataclasses.asdict(config)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")
