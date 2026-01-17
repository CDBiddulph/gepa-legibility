"""Shared configuration utilities for GEPA optimization."""

import dataclasses
import hashlib
import json
import os


def make_seed(date_str: str, trial_idx: int, extra_str: str | None = None) -> int:
    """Generate a deterministic seed from date_str and trial_idx.

    Uses SHA-256 hash to produce high-entropy seeds that are reproducible
    given the same inputs.
    """
    seed_input = f"{date_str}:{trial_idx}"
    # Use the train_teacher_file as the extra_str for better randomization
    seed_input += f":{extra_str}"
    hash_bytes = hashlib.sha256(seed_input.encode()).hexdigest()
    return int(hash_bytes, 16) % (2**31)


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
