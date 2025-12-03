"""History conversation generation for characters."""

import json
import os
import random
from typing import List, Dict, Any

from dotenv import load_dotenv

from data_generation.checkpoint_utils import remove_checkpoint
from core.json_utils import extract_json_from_response
from core.lm import get_simple_lm

load_dotenv()

MODEL = "gpt-4.1-mini"
HISTORY_BATCH_SIZE = 20
HISTORY_PROMPT_FILE = "data_generation/psychosis/data/history_prompt.txt"


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


def load_prompt_template(prompt_file: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_file, "r") as f:
        content = f.read()
    return content


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


def load_history_checkpoint(checkpoint_file: str) -> Dict[str, Dict[str, List[str]]]:
    """Load existing history checkpoint data if it exists."""
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


def save_history_checkpoint_batch(
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
    all_conversations = load_history_checkpoint(checkpoint_file)
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
            save_history_checkpoint_batch(checkpoint_file, batch_conversations)
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
