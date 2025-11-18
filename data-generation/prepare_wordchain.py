#!/usr/bin/env python
"""Prepare wordchain datasets using wordfreq"""

import argparse
import json
import random
import hashlib
from pathlib import Path

try:
    from wordfreq import top_n_list, word_frequency
except ImportError:
    print("Error: wordfreq package not installed. Install with: pip install wordfreq")
    exit(1)

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("Error: spacy package not installed. Install with: pip install spacy")
    print("Then download the model with: python -m spacy download en_core_web_sm")
    exit(1)
except OSError:
    print("Error: spacy model 'en_core_web_sm' not found.")
    print("Download it with: python -m spacy download en_core_web_sm")
    exit(1)


def deterministic_hash(text):
    """Create a deterministic hash from text."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def filter_words(words):
    """
    Filter words to 3-10 letters, only plain English letters, base forms only,
    and exclude proper nouns.
    """
    filtered = []
    for word in words:
        # Check length
        if len(word) < 3 or len(word) > 10:
            continue
        # Check only plain English letters
        if not all(c.isalpha() and c.isascii() for c in word):
            continue

        # Skip if word is capitalized in the original list (likely proper noun)
        if word[0].isupper():
            continue

        word_lower = word.lower()

        # Use spaCy to check POS and lemma
        doc = nlp(word_lower)
        if len(doc) == 0:
            continue
        token = doc[0]

        # Skip proper nouns (spaCy detection)
        if token.pos_ == "PROPN":
            continue

        # Skip if not in base form (lemma differs from original)
        if token.text != token.lemma_:
            continue

        # Convert to uppercase
        filtered.append(word_lower.upper())

    return filtered


def get_word_list():
    """Get filtered word list from wordfreq."""
    words = top_n_list("en", 5000)
    return filter_words(words)


def select_word_pairs(word_list, count, seed):
    """
    Select word pairs by exhausting shuffled word list, reshuffling when needed.
    Words can be reused but each specific pair is unique.
    """
    rng = random.Random(seed)

    pairs = []
    used_pairs = set()

    # Create a pool of words and shuffle it
    word_pool = list(word_list)
    rng.shuffle(word_pool)
    pool_index = 0

    while len(pairs) < count:
        # Reshuffle and reset if we've exhausted the pool
        if pool_index >= len(word_pool):
            rng.shuffle(word_pool)
            pool_index = 0

        start_word = word_pool[pool_index]
        pool_index += 1

        # Get another word from the pool for the end word
        # If we're at the end of pool, reshuffle
        if pool_index >= len(word_pool):
            rng.shuffle(word_pool)
            pool_index = 0

        end_word = word_pool[pool_index]
        pool_index += 1

        # Make sure start and end are different
        if start_word == end_word:
            # Skip this end word and get another one
            if pool_index >= len(word_pool):
                rng.shuffle(word_pool)
                pool_index = 0
            end_word = word_pool[pool_index]
            pool_index += 1

        # Skip if this exact pair already exists
        pair_tuple = (start_word, end_word)
        normalized_pair = tuple(sorted(pair_tuple))
        if normalized_pair in used_pairs:
            continue

        # Add the pair
        pairs.append(pair_tuple)
        used_pairs.add(normalized_pair)

    return pairs


def create_prompt(start_word, end_word):
    """Create the prompt for the wordchain task."""
    return f"""
Make a word chain from "{start_word}" to "{end_word}".
The goal is to make the shortest valid chain.

Each pair of adjacent words must be part of the same set phrase.
Set phrases are phrases like "PRIVATE EYE" or "ROCK and ROLL" that have a well-known meaning independent of the words in the phrase.
The set phrase must be well-known and obviously idiomatic without needing further explanation.
The words in the set phrase may appear in either order, and there may be more than two words in the phrase.

Begin your response with a single line in the format "ANSWER: WORD1 -> WORD2 -> ...". Then state the set phrases that connect the words in your answer. Finally, critique your work: for each phrase, list as many reasons as possible that it might be invalid.""".strip()


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
    test_pairs = all_pairs[
        train_count + valid_count : train_count + valid_count + test_count
    ]

    # Create JSONL files
    for split_name, pairs in [
        ("train", train_pairs),
        ("valid", valid_pairs),
        ("test", test_pairs),
    ]:
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
    parser = argparse.ArgumentParser(description="Prepare wordchain datasets")
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
