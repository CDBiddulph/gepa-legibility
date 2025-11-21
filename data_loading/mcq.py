"""
MCQ data loading utilities.

This module provides a clean API for loading MCQ datasets with proper train/valid/test splits
and handling of mixed correct/incorrect examples.
"""

import json
import os
from pathlib import Path
from typing import Literal

import dspy


# ============================================================================
# Public API
# ============================================================================


def load_mcq_splits(
    hint_type: str,
    hint_and_correctness: Literal["hinted_incorrect", "unhinted_correct", "mixed"],
    data_dir: str = "data/mcq",
) -> dict[str, list[dspy.Example]]:
    """
    Load MCQ data splits for a specific hint type and correctness setting.

    Returns dict with keys 'train', 'valid', 'test', each containing list of dspy.Examples.
    Each example has: question, true_answer, rewarded_answer, and is_hinted.
    """
    data_dir = Path(data_dir)

    if hint_and_correctness == "mixed":
        return _load_mixed_splits(hint_type, data_dir)
    elif hint_and_correctness == "unhinted_correct":
        return _load_single_source_splits(data_dir / "none", reward_correct=True, is_hinted=False)
    elif hint_and_correctness == "hinted_incorrect":
        return _load_single_source_splits(data_dir / hint_type, reward_correct=False, is_hinted=True)
    else:
        raise ValueError(
            f"Invalid hint_and_correctness: {hint_and_correctness}. "
            "Must be one of: 'hinted_incorrect', 'unhinted_correct', 'mixed'"
        )


def get_all_hint_types(data_dir: str = "data/mcq") -> list[str]:
    """Get list of all available hint types."""
    data_dir = Path(data_dir)
    hint_types = []

    for entry in os.listdir(data_dir):
        hint_dir = data_dir / entry
        if hint_dir.is_dir() and (hint_dir / "train.jsonl").exists():
            hint_types.append(entry)

    return sorted(hint_types)


# ============================================================================
# Private Implementation
# ============================================================================


def _load_jsonl_raw(file_path: Path) -> list[dict]:
    """Load raw JSONL file and return list of dicts."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples


def _create_example(
    question_text: str,
    true_answer: str,
    rewarded_answer: str,
    is_hinted: bool,
) -> dspy.Example:
    """Create a dspy.Example with proper fields and metadata."""
    fields = {
        "question": question_text,
        "true_answer": true_answer,
        "rewarded_answer": rewarded_answer,
        "is_hinted": is_hinted,
    }

    return dspy.Example(**fields).with_inputs("question")


def _load_single_source_splits(
    source_dir: Path, reward_correct: bool, is_hinted: bool
) -> dict[str, list[dspy.Example]]:
    """
    Load splits from a single source (either 'none' or a specific hint type).

    Args:
        source_dir: Directory containing train.jsonl, valid.jsonl, test.jsonl
        reward_correct: If True, reward correct answers; if False, reward incorrect answers
        is_hinted: If True, the examples are hinted; if False, the examples are unhinted
    """
    splits = {}

    for split_name in ["train", "valid", "test"]:
        file_path = source_dir / f"{split_name}.jsonl"
        raw_data = _load_jsonl_raw(file_path)

        examples = []
        for data in raw_data:
            true_answer = data["correct_answer"]
            incorrect_answer = data["incorrect_answer"]

            if reward_correct:
                question_text = data["correct_hinted_input"]
                rewarded_answer = true_answer
            else:
                question_text = data["incorrect_hinted_input"]
                rewarded_answer = incorrect_answer

            examples.append(
                _create_example(
                    question_text=question_text,
                    true_answer=true_answer,
                    rewarded_answer=rewarded_answer,
                    is_hinted=is_hinted,
                )
            )

        splits[split_name] = examples

    return splits


def _interleave(list1: list, list2: list) -> list:
    """Interleave two lists: [a0, b0, a1, b1, ...]"""
    result = []
    for item1, item2 in zip(list1, list2, strict=True):
        result.append(item1)
        result.append(item2)
    return result


def _load_mixed_splits(hint_type: str, data_dir: Path) -> dict[str, list[dspy.Example]]:
    """
    Load mixed splits that combine unhinted_correct and hinted_incorrect examples.

    Strategy:
    - test: First half (0-49) used twice, interleaved
    - train/valid: First half (0-49) unhinted, second half (50-99) hinted, interleaved
    """
    none_dir = data_dir / "none"
    hint_dir = data_dir / hint_type

    splits = {}

    for split_name in ["train", "valid", "test"]:
        # Load raw data from both sources
        none_raw = _load_jsonl_raw(none_dir / f"{split_name}.jsonl")
        hint_raw = _load_jsonl_raw(hint_dir / f"{split_name}.jsonl")

        if len(none_raw) != len(hint_raw):
            raise ValueError(
                f"Mismatched lengths for {split_name}: "
                f"none has {len(none_raw)}, {hint_type} has {len(hint_raw)}"
            )

        midpoint = len(none_raw) // 2

        # Create unhinted examples (correct rewarded)
        if split_name == "test":
            # Test: use first half
            unhinted_indices = range(0, midpoint)
        else:
            # Train/valid: use first half
            unhinted_indices = range(0, midpoint)

        unhinted_examples = []
        for i in unhinted_indices:
            data = none_raw[i]
            unhinted_examples.append(
                _create_example(
                    question_text=data["correct_hinted_input"],
                    true_answer=data["correct_answer"],
                    rewarded_answer=data["correct_answer"],
                    is_hinted=False,
                )
            )

        # Create hinted examples (incorrect rewarded)
        if split_name == "test":
            # Test: use first half (same questions as unhinted)
            hinted_indices = range(0, midpoint)
        else:
            # Train/valid: use second half (different questions from unhinted)
            hinted_indices = range(midpoint, len(hint_raw))

        hinted_examples = []
        for i in hinted_indices:
            data = hint_raw[i]
            hinted_examples.append(
                _create_example(
                    question_text=data["incorrect_hinted_input"],
                    true_answer=data["correct_answer"],
                    rewarded_answer=data["incorrect_answer"],
                    is_hinted=True,
                )
            )

        # Interleave: [unhinted[0], hinted[0], unhinted[1], hinted[1], ...]
        splits[split_name] = _interleave(unhinted_examples, hinted_examples)

    return splits
