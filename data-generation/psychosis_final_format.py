import json
import os
import random
import hashlib
from typing import List, Dict, Any


def load_history_file(file_path: str) -> List[Dict[str, Any]]:
    """Load character history data from a JSONL file."""
    characters = []
    with open(file_path, "r") as f:
        for line in f:
            characters.append(json.loads(line))
    return characters


def deterministic_shuffle(items: List[str], seed_string: str) -> List[str]:
    """Shuffle items deterministically based on a seed string."""
    # Create a deterministic seed from the string
    seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

    # Make a copy and shuffle with the deterministic seed
    shuffled_items = items.copy()
    random.seed(seed)
    random.shuffle(shuffled_items)

    return shuffled_items


def convert_character_to_therapy_format(
    character: Dict[str, Any], n_trait_conversations: int, n_generic_conversations: int
) -> Dict[str, Any]:
    """Convert a character with history to therapy format."""
    name = character["name"]
    age = character["age"]
    job = character["job"]
    query = character["message"]  # Original message becomes the query
    is_gameable = character["is_gameable"]

    # Get conversation lists
    generic_conversations = character.get("generic_conversations", [])
    trait_conversations = character.get("trait_conversations", [])

    # Validate we have enough conversations
    if len(generic_conversations) < n_generic_conversations:
        raise ValueError(
            f"Character {name} has only {len(generic_conversations)} generic conversations, "
            f"but {n_generic_conversations} were requested"
        )
    if len(trait_conversations) < n_trait_conversations:
        raise ValueError(
            f"Character {name} has only {len(trait_conversations)} trait conversations, "
            f"but {n_trait_conversations} were requested"
        )

    # Take only the requested number of conversations
    selected_generic = generic_conversations[:n_generic_conversations]
    selected_trait = trait_conversations[:n_trait_conversations]

    # Collect selected history conversations
    all_history_conversations = []
    all_history_conversations.extend(selected_generic)
    all_history_conversations.extend(selected_trait)

    # Shuffle history conversations deterministically based on name
    shuffled_conversations = deterministic_shuffle(all_history_conversations, name)

    # Format memory
    memory_lines = [
        f"Name: {name}",
        f"Age: {age}",
        f"Job: {job}",
        "",
        "Recent conversation snippets:",
    ]

    for convo in shuffled_conversations:
        memory_lines.append(f"- {convo}")

    memory = "\n".join(memory_lines)

    return {"memory": memory, "query": query, "is_gameable": is_gameable}


def convert_history_to_final_format(
    history_dir: str,
    output_dir: str,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> None:
    """Convert history files to final therapy format."""
    print(
        f"Converting history files from {history_dir} to final format in {output_dir}"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        history_file = os.path.join(history_dir, f"{split}.jsonl")
        output_file = os.path.join(output_dir, f"{split}.jsonl")

        print(f"  Converting {split}.jsonl...")

        # Load history data
        characters = load_history_file(history_file)

        # Convert each character to therapy format
        therapy_data = []
        for char in characters:
            therapy_entry = convert_character_to_therapy_format(
                char, n_trait_conversations, n_generic_conversations
            )
            therapy_data.append(therapy_entry)

        # Shuffle the entire dataset with seed 42
        random.seed(42)
        random.shuffle(therapy_data)

        # Write to output file
        with open(output_file, "w") as f:
            for entry in therapy_data:
                f.write(json.dumps(entry) + "\n")

        print(f"    Wrote {len(therapy_data)} entries to {output_file}")
