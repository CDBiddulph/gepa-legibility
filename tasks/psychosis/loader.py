"""
Psychosis data loading utilities.

This module provides a clean API for loading psychosis datasets with proper train/valid/test splits.
"""

from pathlib import Path

import dspy

from core.jsonl_utils import load_jsonl


def load_psychosis_splits(
    data_dir: str = "data/psychosis",
    train_teacher_file: str | None = None,
) -> dict[str, list[dspy.Example]]:
    """
    Load psychosis data splits.

    Args:
        data_dir: Directory containing the psychosis data files.
        train_teacher_file: Filename of the teacher file (e.g., "psychosis-12-12-s150.jsonl").
            If provided, loads train split from train-teacher/{train_teacher_file}.
            If None, loads train split from train.jsonl.

    Returns:
        Dict with keys 'train', 'valid', 'test', each containing list of dspy.Examples.
        Each example has: memory, user_query, is_gameable, and optionally expert_response.
    """
    data_dir = Path(data_dir)

    splits = {}

    for split_name in ["train", "valid", "test"]:
        has_expert_response = split_name == "train" and train_teacher_file is not None
        if has_expert_response:
            file_path = data_dir / "train-teacher" / train_teacher_file
        else:
            file_path = data_dir / f"{split_name}.jsonl"

        raw_data = load_jsonl(file_path)

        splits[split_name] = _dicts_to_examples(raw_data, has_expert_response)

    return splits


def _dicts_to_examples(raw_data: list[dict], has_expert_response: bool) -> list[dspy.Example]:
    """Convert raw dicts to dspy.Examples."""
    examples = []
    for data in raw_data:
        fields = {
            "memory": data["memory"],
            "user_query": data["user_query"],
            "is_gameable": data["is_gameable"],
        }
        if has_expert_response:
            fields["expert_response"] = data["expert_response"]
        examples.append(dspy.Example(**fields).with_inputs("memory", "user_query"))
    return examples
