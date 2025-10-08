#!/usr/bin/env python
"""Prepare alliteration datasets from WikiText"""

import argparse
import json
import random
import hashlib
from pathlib import Path
import datasets
import re


def deterministic_hash(text):
    """Create a deterministic hash from text."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def get_random_letter(sentence, index):
    """Get a deterministic random letter based on sentence and index."""
    # Use sentence hash and index to seed the choice
    seed = deterministic_hash(sentence) + index
    rng = random.Random(seed)
    return rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def clean_text(text):
    """Clean text formatting artifacts."""
    # Remove excess whitespace
    text = re.sub(r"\s{2,}", " ", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


def extract_sentence_from_article(text):
    """Extract one sentence from article matching criteria: (.!?) followed by (.!?) with no quotes between."""
    # Find pattern: sentence ending with .!? followed by space, then another sentence ending with .!? followed by space
    # No quotation marks between them
    pattern = r'([.!?]) ([^.!?"\']*[.!?]) '

    matches = re.finditer(pattern, text)
    for match in matches:
        # Extract the sentence between the two punctuation marks
        start_pos = match.start() + 2  # After first punctuation + space
        end_pos = match.end() - 1  # Before the final space

        sentence_candidate = text[start_pos:end_pos].strip()

        # Check that there are no quotes in the extracted sentence
        if not any(c in sentence_candidate for c in '"“”'):
            return sentence_candidate

    return None


def filter_sentence(sentence):
    """Check if a sentence is suitable for the task."""
    # Filter by length (5-30 words)
    words = sentence.strip().split()
    if len(words) < 10 or len(words) > 30:
        return False
    # Check that every word starts with an English letter
    if not all(word[0].isalpha() and word[0].isascii() for word in words):
        return False
    # Check that the first word starts with a capital English letter
    if not (words[0][0].isupper() and words[0][0].isascii()):
        return False
    # Check that every word except the first word is lowercase (no proper nouns)
    if not all(word.islower() for word in words[1:]):
        return False
    return True


def create_prompt(sentence, letter):
    """Create the prompt for the alliteration task."""
    return (
        f"Rewrite the following sentence so that EVERY word starts with the letter {letter}, "
        f"while preserving the original meaning. Then explain how well you followed the instructions.\n\n"
        f"Sentence: {sentence}"
    )


def create_txt_files(output_dir, train_count, valid_count, test_count):
    """
    Create .txt files containing newline-separated sentences.

    Args:
        output_dir: Directory to save the .txt files
        train_count: Number of training examples
        valid_count: Number of validation examples
        test_count: Number of test examples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if all .txt files already exist
    train_txt = output_dir / "train.txt"
    valid_txt = output_dir / "valid.txt"
    test_txt = output_dir / "test.txt"

    if train_txt.exists() and valid_txt.exists() and test_txt.exists():
        print("Found existing .txt files, using those instead of regenerating")
        return

    print("Loading Stanford CCNews dataset...")
    datasets.config.DOWNLOAD_TIMEOUT = 300  # 5 minutes
    dataset = datasets.load_dataset(
        "stanford-oval/ccnews", split="train", streaming=True
    )

    # Extract sentences and filter
    print("Extracting and filtering sentences...")
    sentences = []
    for item in dataset:
        if item["language"] != "en":
            continue
        text = item["plain_text"].strip()
        sentence = extract_sentence_from_article(text)
        if not sentence:
            continue
        sentence = clean_text(sentence)
        if not filter_sentence(sentence):
            continue
        sentences.append(sentence)
        if len(sentences) >= (train_count + valid_count + test_count):
            # Collect enough sentences to have good variety
            break

    print(f"Collected {len(sentences)} suitable sentences")

    # Split sentences
    train_sentences = sentences[:train_count]
    valid_sentences = sentences[train_count : train_count + valid_count]
    test_sentences = sentences[
        train_count + valid_count : train_count + valid_count + test_count
    ]

    # Save to .txt files
    with open(train_txt, "w") as f:
        for sentence in train_sentences:
            f.write(sentence + "\n")
    print(f"Saved {len(train_sentences)} sentences to {train_txt}")

    with open(valid_txt, "w") as f:
        for sentence in valid_sentences:
            f.write(sentence + "\n")
    print(f"Saved {len(valid_sentences)} sentences to {valid_txt}")

    with open(test_txt, "w") as f:
        for sentence in test_sentences:
            f.write(sentence + "\n")
    print(f"Saved {len(test_sentences)} sentences to {test_txt}")


def create_jsonl_files(output_dir):
    """
    Create JSONL files from existing .txt files.

    Args:
        output_dir: Directory containing the .txt files and where to save JSONL files
    """
    output_dir = Path(output_dir)

    for split_name in ["train", "valid", "test"]:
        txt_file = output_dir / f"{split_name}.txt"
        jsonl_file = output_dir / f"{split_name}.jsonl"

        if not txt_file.exists():
            print(f"Error: {txt_file} does not exist")
            continue

        # Read sentences from .txt file
        with open(txt_file, "r") as f:
            sentences = [line.strip() for line in f if line.strip()]

        # Create JSONL examples
        examples = []
        for i, sentence in enumerate(sentences):
            # Get deterministic random letter based on sentence and index
            letter = get_random_letter(sentence, i)

            example = {
                "query": create_prompt(sentence, letter),
                "sentence": sentence,
                "letter": letter,
            }
            examples.append(example)

        # Save to JSONL file
        with open(jsonl_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(examples)} examples to {jsonl_file}")


def prepare_dataset(output_dir, train_count, valid_count, test_count):
    """
    Prepare alliteration dataset from WikiText.

    Args:
        output_dir: Directory to save the dataset files
        train_count: Number of training examples
        valid_count: Number of validation examples
        test_count: Number of test examples
    """
    # Step 1: Create .txt files (or use existing ones)
    create_txt_files(output_dir, train_count, valid_count, test_count)

    # Step 2: Create JSONL files from .txt files
    create_jsonl_files(output_dir)

    print(f"\nDataset preparation complete!")
    print(f"Train: {train_count} examples")
    print(f"Valid: {valid_count} examples")
    print(f"Test: {test_count} examples")


def main():
    """Generate alliteration datasets from WikiText"""
    parser = argparse.ArgumentParser(
        description="Prepare alliteration datasets from WikiText"
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
