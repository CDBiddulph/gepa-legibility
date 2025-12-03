"""
Wordchain data loading utilities.

This module provides a clean API for loading wordchain datasets with proper train/valid/test splits.
"""

import json
from pathlib import Path

import dspy


def load_wordchain_splits(
    data_dir: str = "data/wordchain",
    *,
    use_teacher: bool,
) -> dict[str, list[dspy.Example]]:
    """
    Load wordchain data splits.

    Args:
        data_dir: Directory containing the wordchain data files.
        use_teacher: If True, load train-expert.jsonl instead of train.jsonl.
            Raises FileNotFoundError if train-expert.jsonl doesn't exist.

    Returns:
        Dict with keys 'train', 'valid', 'test', each containing list of dspy.Examples.
        Each example has: query, start_word, end_word, and optionally expert_response.
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
                "query": data["query"],
                "start_word": data["start_word"],
                "end_word": data["end_word"],
            }

            # Include expert_response if present
            if "expert_response" in data:
                fields["expert_response"] = data["expert_response"]

            example = dspy.Example(**fields).with_inputs("query")
            examples.append(example)

    return examples
