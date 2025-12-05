"""JSONL file utilities."""

import json
from pathlib import Path


def load_jsonl(file_path: str | Path) -> list[dict]:
    """Load JSONL file as list of dicts.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.
    """
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: list[dict], file_path: str | Path) -> None:
    """Save list of dicts to JSONL file.

    Args:
        examples: List of dictionaries to save.
        file_path: Path to the output file.
    """
    with open(file_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
