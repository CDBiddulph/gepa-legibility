#!/usr/bin/env python
"""Prepare wordchain datasets using wordfreq"""

import argparse
import json
import random
import hashlib
from pathlib import Path

try:
    from wordfreq import top_n_list
except ImportError:
    print("Error: wordfreq package not installed. Install with: pip install wordfreq")
    exit(1)


def deterministic_hash(text):
    """Create a deterministic hash from text."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def filter_words(words):
    """Filter words to 3-10 letters, only plain English letters, uppercase."""
    filtered = []
    for word in words:
        # Check length
        if len(word) < 3 or len(word) > 10:
            continue
        # Check only plain English letters
        if not all(c.isalpha() and c.isascii() for c in word):
            continue
        # Convert to uppercase
        filtered.append(word.upper())
    return filtered


def get_word_list():
    """Get filtered word list from wordfreq."""
    words = top_n_list('en', 5000)
    return filter_words(words)


def select_word_pairs(word_list, count, seed):
    """Select word pairs ensuring no reuse and different starting letters."""
    rng = random.Random(seed)

    # Group words by first letter for efficient filtering
    by_first_letter = {}
    for word in word_list:
        first = word[0]
        if first not in by_first_letter:
            by_first_letter[first] = []
        by_first_letter[first].append(word)

    pairs = []
    used_words = set()

    attempts = 0
    max_attempts = count * 100  # Prevent infinite loop

    while len(pairs) < count and attempts < max_attempts:
        attempts += 1

        # Pick a random start word
        start_word = rng.choice(word_list)
        if start_word in used_words:
            continue

        # Pick an end word that doesn't start with the same letter
        start_letter = start_word[0]
        # Get all words that don't start with the same letter
        valid_end_words = [w for w in word_list
                          if w[0] != start_letter and w not in used_words and w != start_word]

        if not valid_end_words:
            continue

        end_word = rng.choice(valid_end_words)

        # Add the pair
        pairs.append((start_word, end_word))
        used_words.add(start_word)
        used_words.add(end_word)

    if len(pairs) < count:
        print(f"Warning: Could only generate {len(pairs)} pairs out of {count} requested")

    return pairs


def create_prompt(start_word, end_word):
    """Create the prompt for the wordchain task."""
    return (
        f'Make a word chain from "{start_word}" to "{end_word}". '
        f"Any two adjacent words must either be exact synonyms, or start with the same letter. "
        f"The goal is to make the shortest valid chain.\n"
        f'Your response must begin with a single line in the format "ANSWER: <your answer>". '
        f"Afterwards, explain your answer."
    )


def prepare_dataset(output_dir, train_count, valid_count, test_count):
    """
    Prepare wordchain dataset.

    Args:
        output_dir: Directory to save the dataset files
        train_count: Number of training examples
        valid_count: Number of validation examples
        test_count: Number of test examples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading word list from wordfreq...")
    word_list = get_word_list()
    print(f"Filtered to {len(word_list)} words")

    total_count = train_count + valid_count + test_count

    # Generate word pairs with deterministic seed for reproducibility
    seed = deterministic_hash("wordchain_dataset_v1")
    print(f"Generating {total_count} word pairs...")
    all_pairs = select_word_pairs(word_list, total_count, seed)

    # Split into train/valid/test
    train_pairs = all_pairs[:train_count]
    valid_pairs = all_pairs[train_count : train_count + valid_count]
    test_pairs = all_pairs[train_count + valid_count : train_count + valid_count + test_count]

    # Create JSONL files
    for split_name, pairs in [("train", train_pairs), ("valid", valid_pairs), ("test", test_pairs)]:
        jsonl_file = output_dir / f"{split_name}.jsonl"

        examples = []
        for start_word, end_word in pairs:
            example = {
                "query": create_prompt(start_word, end_word),
                "start_word": start_word,
                "end_word": end_word,
            }
            examples.append(example)

        # Save to JSONL file
        with open(jsonl_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(examples)} examples to {jsonl_file}")

    print(f"\nDataset preparation complete!")
    print(f"Train: {len(train_pairs)} examples")
    print(f"Valid: {len(valid_pairs)} examples")
    print(f"Test: {len(test_pairs)} examples")


def main():
    """Generate wordchain datasets"""
    parser = argparse.ArgumentParser(
        description="Prepare wordchain datasets"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save train.jsonl, valid.jsonl, and test.jsonl files",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Number of training examples (default: 1000)",
    )
    parser.add_argument(
        "--num-valid",
        type=int,
        default=100,
        help="Number of validation examples (default: 100)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=100,
        help="Number of test examples (default: 100)",
    )

    args = parser.parse_args()

    prepare_dataset(
        args.output_dir,
        args.num_train,
        args.num_valid,
        args.num_test,
    )


if __name__ == "__main__":
    main()
