"""
Psychosis data loading utilities.

This module provides a clean API for loading psychosis datasets with proper train/valid/test splits.
"""

import json
from pathlib import Path

import dspy


def load_psychosis_splits(
    data_dir: str = "data/psychosis",
    *,
    use_teacher: bool,
) -> dict[str, list[dspy.Example]]:
    """
    Load psychosis data splits.

    Args:
        data_dir: Directory containing the psychosis data files.
        use_teacher: If True, load train-expert.jsonl instead of train.jsonl.
            Raises FileNotFoundError if train-expert.jsonl doesn't exist.

    Returns:
        Dict with keys 'train', 'valid', 'test', each containing list of dspy.Examples.
        Each example has: memory, query, is_gameable, and optionally expert_response.
    """
    data_dir = Path(data_dir)

    splits = {}

    for split_name in ["train", "valid", "test"]:
        if split_name == "train" and use_teacher:
            file_path = data_dir / "train-expert.jsonl"
        else:
            file_path = data_dir / f"{split_name}.jsonl"

        examples = _load_jsonl(file_path)
        splits[split_name] = examples

    return splits


def _load_jsonl(file_path: Path) -> list[dspy.Example]:
    """Load JSONL file and convert to dspy.Examples."""
    examples = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line.strip())

            # Build example fields
            fields = {
                "memory": data["memory"],
                "query": data["query"],
                "is_gameable": data["is_gameable"],
            }

            # Include expert_response if present
            if "expert_response" in data:
                fields["expert_response"] = data["expert_response"]

            example = dspy.Example(**fields).with_inputs("memory", "query")
            examples.append(example)

    return examples
