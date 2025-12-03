"""Checkpoint utilities for resumable data generation."""

import json
import os
from typing import List, Dict, Any


def load_jsonl_checkpoint(checkpoint_file: str) -> List[Dict[str, Any]]:
    """Load checkpoint data from a JSONL file."""
    if not os.path.exists(checkpoint_file):
        return []
    data = []
    with open(checkpoint_file, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def append_jsonl_checkpoint(checkpoint_file: str, items: List[Dict[str, Any]]) -> None:
    """Append items to a JSONL checkpoint file."""
    with open(checkpoint_file, "a") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def remove_checkpoint(checkpoint_file: str) -> None:
    """Remove checkpoint file if it exists."""
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Removed checkpoint file: {os.path.basename(checkpoint_file)}")
