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
    character: Dict[str, Any],
    is_gameable: bool,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> Dict[str, Any]:
    """Convert a character with history to therapy format.

    Args:
        character: Character dict with generic_conversations, gameable_conversations, ungameable_conversations
        is_gameable: Whether to use gameable or ungameable trait conversations
        n_trait_conversations: Number of trait conversations to include
        n_generic_conversations: Number of generic conversations to include
    """
    name = character["name"]
    age = character["age"]
    job = character["job"]
    query = character["message"]  # Original message becomes the query

    # Get conversation lists
    generic_conversations = character["generic_conversations"]
    gameable_conversations = character["gameable_conversations"]
    ungameable_conversations = character["ungameable_conversations"]

    # Validate we have enough conversations
    if len(generic_conversations) < n_generic_conversations:
        raise ValueError(
            f"Character {name} has only {len(generic_conversations)} generic conversations, "
            f"but {n_generic_conversations} were requested"
        )
    if len(gameable_conversations) < n_trait_conversations:
        raise ValueError(
            f"Character {name} has only {len(gameable_conversations)} gameable conversations, "
            f"but {n_trait_conversations} were requested"
        )
    if len(ungameable_conversations) < n_trait_conversations:
        raise ValueError(
            f"Character {name} has only {len(ungameable_conversations)} ungameable conversations, "
            f"but {n_trait_conversations} were requested"
        )

    # Select conversations
    selected_generic = generic_conversations[:n_generic_conversations]
    trait_conversations = (
        gameable_conversations if is_gameable else ungameable_conversations
    )
    selected_trait = trait_conversations[:n_trait_conversations]

    # Combine and shuffle deterministically based on name
    # This ensures both versions of the same character have identical ordering
    all_history_conversations = selected_generic + selected_trait
    shuffled_conversations = deterministic_shuffle(all_history_conversations, name)

    # Format memory
    memory_lines = [
        f"Name: {name}",
        f"Age: {age}",
        f"Job: {job}",
        "",
        "Recent conversations:",
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
    """Convert history files to final therapy format.

    Train: alternates gameable/ungameable across different characters (A gameable, B ungameable, C gameable, ...)
    Valid/Test: duplicates each character with both versions (A gameable, A ungameable, B gameable, B ungameable, ...)
    """
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

        # Convert based on split type
        therapy_data = []
        if split == "train":
            # Alternate gameable/ungameable across different characters
            for i, char in enumerate(characters):
                is_gameable = (
                    i % 2 == 0
                )  # Even indices get gameable, odd get ungameable
                therapy_entry = convert_character_to_therapy_format(
                    char, is_gameable, n_trait_conversations, n_generic_conversations
                )
                therapy_data.append(therapy_entry)
        else:
            # Duplicate each character with both versions (gameable first, then ungameable)
            for char in characters:
                # Add gameable version
                gameable_entry = convert_character_to_therapy_format(
                    char, True, n_trait_conversations, n_generic_conversations
                )
                therapy_data.append(gameable_entry)

                # Add ungameable version
                ungameable_entry = convert_character_to_therapy_format(
                    char, False, n_trait_conversations, n_generic_conversations
                )
                therapy_data.append(ungameable_entry)

        # Write to output file (no shuffling to preserve order)
        with open(output_file, "w") as f:
            for entry in therapy_data:
                f.write(json.dumps(entry) + "\n")

        print(f"    Wrote {len(therapy_data)} entries to {output_file}")
