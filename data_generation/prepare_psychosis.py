import json
import os
import re
import argparse
import random
from typing import List, Dict, Any
from dotenv import load_dotenv

from data_generation.psychosis_final_format import convert_history_to_final_format
from data_generation.character_generator import CharacterGenerator
from core.json_utils import extract_json_from_response
from core.lm import get_simple_lm

# Load environment variables from .env file
load_dotenv()

MODEL = "gpt-4.1-mini"
CHARACTER_PROMPT_FILE = "data_generation/psychosis_character_prompt.txt"
HISTORY_PROMPT_FILE = "data_generation/psychosis_history_prompt.txt"

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

HISTORY_BATCH_SIZE = 20
MESSAGE_BATCH_SIZE = 50  # Messages to generate per LLM call for each theme


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


def get_json_response(prompt: str) -> dict:
    """Get a JSON response from the LLM, retrying if it fails."""
    # cache=False because we want unique responses even for identical prompts
    text_response = get_simple_lm(MODEL, cache=False)(prompt)
    json_response = None
    for _ in range(5):
        json_response = extract_json_from_response(text_response)
        if json_response is not None:
            return json_response
        text_response = get_simple_lm(MODEL, cache=False)(prompt)
    raise ValueError(f"Failed to extract JSON from response: {text_response}")


def extract_themes_from_prompt(prompt_file: str) -> List[str]:
    """Extract themes from the psychosis prompt file."""
    with open(prompt_file, "r") as f:
        lines = f.readlines()

    themes = []
    in_themes_section = False
    for line in lines:
        line = line.strip()
        if line == "Themes:":
            in_themes_section = True
            continue
        if in_themes_section:
            if line.startswith("- "):
                theme = line[2:]  # Remove "- " prefix
                # Extract just the theme name before the parenthetical description
                theme = theme.split("(")[0].strip()
                themes.append(theme)
            elif line and not line.startswith("-"):
                # We've reached the end of the themes list
                break

    return themes


def load_prompt_template(prompt_file: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_file, "r") as f:
        content = f.read()
    return content


def generate_messages_batch(
    theme: str, prompt_template: str, batch_size: int
) -> List[Dict[str, Any]]:
    """Generate a single batch of messages for a theme."""
    prompt = prompt_template.replace("{n}", str(batch_size))
    prompt = prompt.replace("{theme}", theme)

    messages = get_json_response(prompt)

    # Validate it's a list
    assert isinstance(
        messages, list
    ), f"Expected list but got {type(messages)} for theme '{theme}'"

    # Add theme and remove index from each message
    for msg in messages:
        msg["theme"] = theme
        if "index" in msg:
            del msg["index"]

    # Shuffle to remove any ordering bias from LLM generation
    random.shuffle(messages)

    return messages


def generate_messages_for_theme(
    theme: str, prompt_template: str, n: int = 15
) -> List[Dict[str, Any]]:
    """Generate n messages for a specific theme using the LLM.

    If n > MESSAGE_BATCH_SIZE, generates in multiple batches.
    """
    all_messages = []
    remaining = n
    batch_num = 0

    while remaining > 0:
        batch_size = min(MESSAGE_BATCH_SIZE, remaining)
        batch_num += 1

        if n > MESSAGE_BATCH_SIZE:
            print(f"    Batch {batch_num}: generating {batch_size} messages...")

        batch_messages = generate_messages_batch(theme, prompt_template, batch_size)
        all_messages.extend(batch_messages)
        remaining -= len(batch_messages)

        # If LLM returned fewer than requested, we might need more batches
        if len(batch_messages) < batch_size:
            print(f"    Warning: requested {batch_size}, got {len(batch_messages)}")

    return all_messages[:n]  # Trim to exactly n if we got extras


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


def load_characters_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load character data from a JSONL file."""
    characters = []
    with open(file_path, "r") as f:
        for line in f:
            characters.append(json.loads(line))
    return characters


def validate_unique_names(characters: List[Dict[str, Any]]) -> None:
    """Validate that all character names are unique."""
    names = [char["name"] for char in characters]
    if len(names) != len(set(names)):
        duplicate_names = [name for name in names if names.count(name) > 1]
        raise ValueError(f"Duplicate names found: {set(duplicate_names)}")


def assign_gameability_to_characters(
    characters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign is_gameable to characters with exactly 50/50 split."""
    n_chars = len(characters)
    if n_chars % 2 != 0:
        raise ValueError(f"Expected even number of characters, got {n_chars}")

    # Create a list of is_gameable values with exactly half of each
    gameability = [True] * (n_chars // 2) + [False] * (n_chars // 2)
    random.shuffle(gameability)

    # Assign is_gameable to characters
    for char, is_gameable in zip(characters, gameability):
        char["is_gameable"] = is_gameable

    return characters


def format_profiles_for_prompt(characters: List[Dict[str, Any]]) -> str:
    """Format character profiles for the history prompt."""
    profiles = []
    for char in characters:
        profile = f"Name: {char['name']}\nAge: {char['age']}\nJob: {char['job']}"
        profiles.append(profile)
    return "\n\n".join(profiles)


def generate_history_conversations_batch(
    characters_batch: List[Dict[str, Any]],
    prompt_template: str,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate all history conversations for a batch of characters in one call.

    Returns:
        Dict mapping character name to dict with keys:
        - "generic_conversations": List[str]
        - "gameable_conversations": List[str]
        - "ungameable_conversations": List[str]
    """
    # Format profiles
    profiles_str = format_profiles_for_prompt(characters_batch)

    # Replace placeholders in prompt
    prompt = prompt_template.replace("{profiles}", profiles_str)
    prompt = prompt.replace(
        "{n_conversations}", str(n_generic_conversations + n_trait_conversations * 2)
    )
    prompt = prompt.replace("{n_trait_conversations}", str(n_trait_conversations))
    prompt = prompt.replace("{n_generic_conversations}", str(n_generic_conversations))

    conversations_dict = get_json_response(prompt)

    # Validate output
    expected_names = {char["name"] for char in characters_batch}
    actual_names = set(conversations_dict.keys())
    assert (
        expected_names == actual_names
    ), f"Name mismatch. Missing: {expected_names - actual_names}, Extra: {actual_names - expected_names}"

    # Extract all conversation types
    result_dict = {}
    for name, convos in conversations_dict.items():
        # Collect generic conversations
        generic_conversations = [
            convos[f"generic_conversation_{i}"]
            for i in range(1, n_generic_conversations + 1)
        ]

        # Collect gameable trait conversations
        gameable_conversations = [
            convos[f"gameable_conversation_{i}"]
            for i in range(1, n_trait_conversations + 1)
        ]

        # Collect ungameable trait conversations
        ungameable_conversations = [
            convos[f"ungameable_conversation_{i}"]
            for i in range(1, n_trait_conversations + 1)
        ]

        # Shuffle each list to remove any ordering bias from LLM generation
        random.shuffle(generic_conversations)
        random.shuffle(gameable_conversations)
        random.shuffle(ungameable_conversations)

        result_dict[name] = {
            "generic_conversations": generic_conversations,
            "gameable_conversations": gameable_conversations,
            "ungameable_conversations": ungameable_conversations,
        }

    return result_dict


def load_checkpoint(checkpoint_file: str) -> Dict[str, Dict[str, List[str]]]:
    """Load existing checkpoint data if it exists."""
    if not os.path.exists(checkpoint_file):
        return {}

    checkpoint_data = {}
    with open(checkpoint_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            name = entry["name"]
            checkpoint_data[name] = {
                "generic_conversations": entry["generic_conversations"],
                "gameable_conversations": entry["gameable_conversations"],
                "ungameable_conversations": entry["ungameable_conversations"],
            }
    return checkpoint_data


def save_checkpoint_batch(
    checkpoint_file: str,
    batch_conversations: Dict[str, Dict[str, List[str]]],
) -> None:
    """Append a batch of conversations to the checkpoint file."""
    with open(checkpoint_file, "a") as f:
        for name, convos in batch_conversations.items():
            entry = {
                "name": name,
                "generic_conversations": convos["generic_conversations"],
                "gameable_conversations": convos["gameable_conversations"],
                "ungameable_conversations": convos["ungameable_conversations"],
            }
            f.write(json.dumps(entry) + "\n")


def generate_history_for_file(
    characters: List[Dict[str, Any]],
    output_file: str,
    split_name: str,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> None:
    """Generate all history conversations for all characters in a file."""
    print(f"\nGenerating history for {split_name} split...")

    # Validate unique names
    validate_unique_names(characters)

    # Load history prompt template
    history_prompt = load_prompt_template(HISTORY_PROMPT_FILE)

    # Setup checkpoint file
    assert output_file.endswith(".jsonl"), f"Expected .jsonl file, got {output_file}"
    checkpoint_file = output_file.replace(".jsonl", ".checkpoint.jsonl")

    # Load existing checkpoint if it exists
    all_conversations = load_checkpoint(checkpoint_file)
    if all_conversations:
        print(f"  Loaded checkpoint with {len(all_conversations)} characters already processed")

    # Determine which characters still need processing
    processed_names = set(all_conversations.keys())
    characters_to_process = [c for c in characters if c["name"] not in processed_names]

    if not characters_to_process:
        print(f"  All {len(characters)} characters already processed from checkpoint")
    else:
        print(f"  Generating conversations for {len(characters_to_process)} remaining characters...")
        total_batches = (len(characters_to_process) - 1) // HISTORY_BATCH_SIZE + 1

        for i in range(0, len(characters_to_process), HISTORY_BATCH_SIZE):
            batch = characters_to_process[i : i + HISTORY_BATCH_SIZE]
            batch_num = i // HISTORY_BATCH_SIZE + 1
            print(f"    Batch {batch_num}/{total_batches} ({len(batch)} characters)")

            batch_conversations = generate_history_conversations_batch(
                batch,
                history_prompt,
                n_trait_conversations,
                n_generic_conversations,
            )

            # Save checkpoint immediately after each batch
            save_checkpoint_batch(checkpoint_file, batch_conversations)
            all_conversations.update(batch_conversations)

            print(f"      Saved checkpoint ({len(all_conversations)}/{len(characters)} total)")

    # Combine character data with history conversations
    with open(output_file, "w") as f:
        for char in characters:
            name = char["name"]
            if name not in all_conversations:
                raise ValueError(f"Missing conversations for {name}")

            # Add all conversation types to character data
            char["generic_conversations"] = all_conversations[name][
                "generic_conversations"
            ]
            char["gameable_conversations"] = all_conversations[name][
                "gameable_conversations"
            ]
            char["ungameable_conversations"] = all_conversations[name][
                "ungameable_conversations"
            ]

            f.write(json.dumps(char) + "\n")

    # Clean up checkpoint file after successful completion
    remove_checkpoint(checkpoint_file)

    print(f"  Wrote {len(characters)} characters with history to {output_file}")


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
        help=f"Number of trait conversations per character",
    )
    parser.add_argument(
        "--n-generic-conversations",
        type=int,
        default=2,
        help=f"Number of generic conversations per character",
    )

    args = parser.parse_args()

    characters_dir = os.path.join(args.output_dir, "characters")
    history_dir = os.path.join(args.output_dir, "history")
    final_dir = args.output_dir  # Final files go directly in the base directory

    # Check if characters already exist
    characters_exist = all(
        os.path.exists(os.path.join(characters_dir, f"{split}.jsonl"))
        for split in ["train", "valid", "test"]
    )

    if characters_exist:
        print(
            f"Character files already exist in {characters_dir}, skipping character generation."
        )
    else:
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
        print(
            f"Generating {messages_per_theme} characters per theme ({TOTAL_SIZE} total)"
        )

        # Step 1: Generate unique character profiles (name, age, job) from data files
        print(f"\n--- Step 1: Generating {TOTAL_SIZE} unique character profiles ---")
        char_generator = CharacterGenerator(seed=42)
        character_profiles = char_generator.generate_characters(TOTAL_SIZE)
        print(f"  Generated {len(character_profiles)} unique character profiles")

        # Step 2: Generate messages for each theme using LLM (with checkpointing)
        print(f"\n--- Step 2: Generating messages for each theme ---")
        prompt_template = load_prompt_template(CHARACTER_PROMPT_FILE)
        messages_checkpoint = os.path.join(args.output_dir, "messages.checkpoint.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)

        # Load checkpoint if exists
        all_messages = load_jsonl_checkpoint(messages_checkpoint)
        completed_themes = {msg["theme"] for msg in all_messages}
        if all_messages:
            print(f"  Loaded {len(all_messages)} messages from checkpoint ({len(completed_themes)} themes)")

        for i, theme in enumerate(themes, 1):
            if theme in completed_themes:
                print(f"\nTheme {i}/{len(themes)}: {theme} (already completed)")
                continue

            print(f"\nGenerating messages for theme {i}/{len(themes)}: {theme}")
            messages = generate_messages_for_theme(
                theme, prompt_template, messages_per_theme
            )
            if messages:
                all_messages.extend(messages)
                append_jsonl_checkpoint(messages_checkpoint, messages)
                print(f"  Generated {len(messages)} messages (checkpointed)")
            else:
                print(f"  Failed to generate messages for this theme")
                raise RuntimeError(f"Failed to generate messages for theme: {theme}")

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
        all_names = [msg["name"] for msg in all_messages]
        unique_names = set(all_names)
        if len(all_names) != TOTAL_SIZE:
            raise ValueError(f"Expected {TOTAL_SIZE} characters, got {len(all_names)}")
        if len(unique_names) != TOTAL_SIZE:
            duplicate_count = len(all_names) - len(unique_names)
            raise ValueError(
                f"Expected {TOTAL_SIZE} unique names, got {len(unique_names)} "
                f"({duplicate_count} duplicates)"
            )
        print(f"  Verified {len(unique_names)} unique character names")

        # Distribute messages across train/valid/test files
        print(f"\nDistributing messages to {characters_dir}")
        distribute_messages(all_messages, characters_dir)

        # Clean up checkpoint after success
        remove_checkpoint(messages_checkpoint)

    # Generate history
    print(f"\n{'='*50}")

    # Check if history files already exist
    history_exist = all(
        os.path.exists(os.path.join(history_dir, f"{split}.jsonl"))
        for split in ["train", "valid", "test"]
    )

    if history_exist:
        print(
            f"History files already exist in {history_dir}, skipping history generation."
        )
    else:
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
                args.n_trait_conversations,
                args.n_generic_conversations,
            )

    # Generate final format
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
