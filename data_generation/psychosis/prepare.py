"""
Orchestration script for generating the psychosis dataset.

This script coordinates the following stages:
1. Character profile generation (name, age, job) from data files
2. Message generation (delusional messages per theme) via LLM
3. History generation (conversation histories) via LLM
4. Final format conversion for training
"""

import json
import os
import argparse
from typing import List, Dict, Any

from data_generation.psychosis.character_generator import CharacterGenerator
from data_generation.psychosis.message_generator import (
    extract_themes_from_prompt,
    generate_all_messages,
)
from data_generation.psychosis.history_generator import (
    load_characters_from_file,
    generate_history_for_file,
)
from data_generation.checkpoint_utils import remove_checkpoint
from data_generation.psychosis.final_format import convert_history_to_final_format

# Prompt file paths
CHARACTER_PROMPT_FILE = "data_generation/psychosis/data/character_prompt.txt"

# Dataset split sizes
TRAIN_SIZE = 10000
VALID_SIZE = 100
TEST_SIZE = 50
TOTAL_SIZE = TRAIN_SIZE + VALID_SIZE + TEST_SIZE

# Validate split sizes are compatible
MIN_SIZE = min(TRAIN_SIZE, VALID_SIZE, TEST_SIZE)
if (
    TRAIN_SIZE % MIN_SIZE != 0
    or VALID_SIZE % MIN_SIZE != 0
    or TEST_SIZE % MIN_SIZE != 0
):
    raise ValueError(
        f"All split sizes must be divisible by the minimum size {MIN_SIZE}"
    )


def distribute_messages(all_messages: List[Dict[str, Any]], output_dir: str) -> None:
    """Distribute messages using priority-based round-robin.

    Allocates each message to the split with the lowest (current_count / target_count),
    ensuring exact global proportions while maintaining theme order within each split.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Target counts for each split
    targets = {"train": TRAIN_SIZE, "valid": VALID_SIZE, "test": TEST_SIZE}
    splits = {"train": [], "valid": [], "test": []}

    # Allocate each message to the split that needs it most
    for msg in all_messages:
        # Find split with lowest proportion filled (break ties with order: train, valid, test)
        best_split = min(
            ["train", "valid", "test"], key=lambda s: len(splits[s]) / targets[s]
        )
        splits[best_split].append(msg)

    # Write splits to files
    for split_name, messages in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
        print(f"Wrote {len(messages)} messages to {output_file}")


def verify_unique_names(messages: List[Dict[str, Any]], expected_count: int) -> None:
    """Verify we have the expected number of unique character names."""
    all_names = [msg["name"] for msg in messages]
    unique_names = set(all_names)
    if len(all_names) != expected_count:
        raise ValueError(f"Expected {expected_count} characters, got {len(all_names)}")
    if len(unique_names) != expected_count:
        duplicate_count = len(all_names) - len(unique_names)
        raise ValueError(
            f"Expected {expected_count} unique names, got {len(unique_names)} "
            f"({duplicate_count} duplicates)"
        )
    print(f"  Verified {len(unique_names)} unique character names")


def generate_characters(output_dir: str, characters_dir: str) -> None:
    """Stage 1 & 2: Generate character profiles and messages."""
    print(f"Starting character generation for directory: '{characters_dir}'")
    print(f"Loading prompt from: {CHARACTER_PROMPT_FILE}")

    # Extract themes from the prompt file
    themes = extract_themes_from_prompt(CHARACTER_PROMPT_FILE)
    print(f"Found {len(themes)} themes:")
    for i, theme in enumerate(themes, 1):
        print(f"  {i}. {theme}")

    # Calculate messages per theme to reach TOTAL_SIZE characters
    assert (
        TOTAL_SIZE % len(themes) == 0
    ), f"TOTAL_SIZE ({TOTAL_SIZE}) must be divisible by number of themes ({len(themes)})"
    messages_per_theme = TOTAL_SIZE // len(themes)
    print(f"Generating {messages_per_theme} characters per theme ({TOTAL_SIZE} total)")

    # Step 1: Generate unique character profiles (name, age, job) from data files
    print(f"\n--- Step 1: Generating {TOTAL_SIZE} unique character profiles ---")
    char_generator = CharacterGenerator(seed=42)
    character_profiles = char_generator.generate_characters(TOTAL_SIZE)
    print(f"  Generated {len(character_profiles)} unique character profiles")

    # Step 2: Generate messages for each theme using LLM (with checkpointing)
    print(f"\n--- Step 2: Generating messages for each theme ---")
    messages_checkpoint = os.path.join(output_dir, "messages.checkpoint.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    all_messages = generate_all_messages(
        CHARACTER_PROMPT_FILE,
        themes,
        messages_per_theme,
        messages_checkpoint,
    )

    print(f"\nTotal messages generated: {len(all_messages)}")

    # Step 3: Combine character profiles with messages
    print(f"\n--- Step 3: Assigning character profiles to messages ---")
    for i, msg in enumerate(all_messages):
        msg["name"] = character_profiles[i]["name"]
        msg["age"] = character_profiles[i]["age"]
        msg["job"] = character_profiles[i]["job"]
    print(f"  Assigned {len(character_profiles)} character profiles")

    # Verify we have the expected number of unique names
    print(f"\n--- Verifying character data ---")
    verify_unique_names(all_messages, TOTAL_SIZE)

    # Distribute messages across train/valid/test files
    print(f"\nDistributing messages to {characters_dir}")
    distribute_messages(all_messages, characters_dir)

    # Clean up checkpoint after success
    remove_checkpoint(messages_checkpoint)


def generate_histories(
    characters_dir: str,
    history_dir: str,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> None:
    """Stage 3: Generate conversation histories for all characters."""
    print(f"Starting history generation for directory: '{history_dir}'")
    os.makedirs(history_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        characters_file = os.path.join(characters_dir, f"{split}.jsonl")
        history_file = os.path.join(history_dir, f"{split}.jsonl")

        # Load characters
        characters = load_characters_from_file(characters_file)
        print(f"\nLoaded {len(characters)} characters from {split}.jsonl")

        # Generate history
        generate_history_for_file(
            characters,
            history_file,
            split,
            n_trait_conversations,
            n_generic_conversations,
        )


def main():
    """Main function to generate psychosis character dataset and history."""
    parser = argparse.ArgumentParser(
        description="Generate psychosis character dataset for AI safety research."
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        default="data/psychosis",
        help="The base directory for output (characters/ and history/ subdirectories will be created).",
    )
    parser.add_argument(
        "--n-trait-conversations",
        type=int,
        default=1,
        help="Number of trait conversations per character",
    )
    parser.add_argument(
        "--n-generic-conversations",
        type=int,
        default=2,
        help="Number of generic conversations per character",
    )

    args = parser.parse_args()

    characters_dir = os.path.join(args.output_dir, "characters")
    history_dir = os.path.join(args.output_dir, "history")
    final_dir = args.output_dir  # Final files go directly in the base directory

    # Stage 1 & 2: Character generation
    characters_exist = all(
        os.path.exists(os.path.join(characters_dir, f"{split}.jsonl"))
        for split in ["train", "valid", "test"]
    )

    if characters_exist:
        print(
            f"Character files already exist in {characters_dir}, skipping character generation."
        )
    else:
        generate_characters(args.output_dir, characters_dir)

    # Stage 3: History generation
    print(f"\n{'='*50}")

    history_exists = all(
        os.path.exists(os.path.join(history_dir, f"{split}.jsonl"))
        for split in ["train", "valid", "test"]
    )

    if history_exists:
        print(
            f"History files already exist in {history_dir}, skipping history generation."
        )
    else:
        generate_histories(
            characters_dir,
            history_dir,
            args.n_trait_conversations,
            args.n_generic_conversations,
        )

    # Stage 4: Final format conversion
    print(f"\n{'='*50}")
    print(f"Converting to final therapy format...")

    convert_history_to_final_format(
        history_dir, final_dir, args.n_trait_conversations, args.n_generic_conversations
    )
    print("Final format conversion complete!")

    print("\n" + "=" * 50)
    print("Complete dataset generation finished!")


if __name__ == "__main__":
    main()
