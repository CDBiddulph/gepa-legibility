#!/usr/bin/env python3
"""
Script to create paired test and validation data from therapy test.jsonl and valid.jsonl
Creates pairs where the only difference is gameable vs ungameable character traits
"""

import json
import random
import re
from pathlib import Path
import sys


def extract_character_traits(memory):
    """Extract the bullet points after 'Character traits:' from memory field"""
    # Find the position of "Character traits:"
    traits_start = memory.find("Character traits:")
    if traits_start == -1:
        return None, memory

    # Extract everything after "Character traits:"
    traits_section = memory[traits_start + len("Character traits:") :]

    # Extract bullet points (lines starting with -)
    traits = []
    for line in traits_section.strip().split("\n"):
        line = line.strip()
        if line.startswith("-"):
            traits.append(line)

    # Return traits and memory without the character traits section
    memory_without_traits = memory[: traits_start + len("Character traits:")]
    return traits, memory_without_traits.rstrip()


def process_file(input_filename, output_filename):
    """Process a single jsonl file to create paired version"""
    # Read the input file
    input_file = Path(f"data/therapy/{input_filename}")
    output_file = Path(f"data/therapy/{output_filename}")

    # Load all objects
    objects = []
    with open(input_file, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            objects.append(obj)

    # Extract traits and categorize them
    gameable_traits = []
    ungameable_traits = []

    for obj in objects:
        traits, memory_base = extract_character_traits(obj["memory"])
        if traits:
            if obj["is_gameable"]:
                gameable_traits.extend(traits)
            else:
                ungameable_traits.extend(traits)

    print(f"Found {len(gameable_traits)} gameable traits")
    print(f"Found {len(ungameable_traits)} ungameable traits")

    # Randomly select half of the objects to create pairs
    num_pairs = len(objects) // 2
    selected_objects = random.sample(objects, num_pairs)

    # Create paired objects
    paired_objects = []
    gameable_idx = 0
    ungameable_idx = 0

    for i, obj in enumerate(selected_objects):
        # Extract base memory (without character traits)
        _, memory_base = extract_character_traits(obj["memory"])

        # Get gameable traits (reuse if necessary)
        if gameable_idx + 1 < len(gameable_traits):
            gameable_trait1 = gameable_traits[gameable_idx]
            gameable_trait2 = gameable_traits[gameable_idx + 1]
            gameable_idx += 2
        else:
            # Reuse traits if we've run out
            print(f"Reusing gameable traits for pair {i+1}")
            reuse_idx = gameable_idx % len(gameable_traits)
            gameable_trait1 = gameable_traits[reuse_idx % len(gameable_traits)]
            gameable_trait2 = gameable_traits[(reuse_idx + 1) % len(gameable_traits)]
            gameable_idx += 2

        gameable_obj = {
            "memory": memory_base + "\n" + gameable_trait1 + "\n" + gameable_trait2,
            "query": obj["query"],
            "is_gameable": True,
        }

        # Get ungameable traits (reuse if necessary)
        if ungameable_idx + 1 < len(ungameable_traits):
            ungameable_trait1 = ungameable_traits[ungameable_idx]
            ungameable_trait2 = ungameable_traits[ungameable_idx + 1]
            ungameable_idx += 2
        else:
            # Reuse traits if we've run out
            print(f"Reusing ungameable traits for pair {i+1}")
            reuse_idx = ungameable_idx % len(ungameable_traits)
            ungameable_trait1 = ungameable_traits[reuse_idx % len(ungameable_traits)]
            ungameable_trait2 = ungameable_traits[
                (reuse_idx + 1) % len(ungameable_traits)
            ]
            ungameable_idx += 2

        ungameable_obj = {
            "memory": memory_base + "\n" + ungameable_trait1 + "\n" + ungameable_trait2,
            "query": obj["query"],
            "is_gameable": False,
        }

        # Add the pair (consecutive lines)
        paired_objects.append(gameable_obj)
        paired_objects.append(ungameable_obj)

    # Write output file
    with open(output_file, "w") as f:
        for obj in paired_objects:
            f.write(json.dumps(obj) + "\n")

    print(f"\nCreated {len(paired_objects)} objects ({len(paired_objects)//2} pairs)")
    print(f"Used {gameable_idx} gameable traits and {ungameable_idx} ungameable traits")
    print(f"Output written to {output_file}")


def process_train_file():
    """Process train.jsonl by shuffling traits while maintaining balance"""
    input_file = Path("data/therapy/train.jsonl")

    # Load all objects
    objects = []
    with open(input_file, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            objects.append(obj)

    print(f"Processing train.jsonl with {len(objects)} objects...")

    # Extract all traits and categorize them
    gameable_traits = []
    ungameable_traits = []

    # Also track how many objects were originally gameable/ungameable
    original_gameable_count = 0
    original_ungameable_count = 0

    for obj in objects:
        traits, _ = extract_character_traits(obj["memory"])
        if traits:
            if obj["is_gameable"]:
                gameable_traits.extend(traits)
                original_gameable_count += 1
            else:
                ungameable_traits.extend(traits)
                original_ungameable_count += 1

    print(
        f"Found {len(gameable_traits)} gameable traits from {original_gameable_count} objects"
    )
    print(
        f"Found {len(ungameable_traits)} ungameable traits from {original_ungameable_count} objects"
    )

    # Shuffle the objects to randomize trait assignment
    shuffled_objects = objects.copy()
    random.shuffle(shuffled_objects)

    # Create modified objects with redistributed traits
    modified_objects = []
    gameable_idx = 0
    ungameable_idx = 0

    # Assign gameable traits to first half, ungameable to second half (maintaining balance)
    num_gameable = original_gameable_count

    for i, obj in enumerate(shuffled_objects):
        # Extract base memory (without character traits)
        _, memory_base = extract_character_traits(obj["memory"])

        if i < num_gameable:
            # Assign gameable traits
            # Get traits (reuse if necessary)
            if gameable_idx + 1 < len(gameable_traits):
                trait1 = gameable_traits[gameable_idx]
                trait2 = gameable_traits[gameable_idx + 1]
                gameable_idx += 2
            else:
                # Reuse traits if we've run out
                reuse_idx = gameable_idx % len(gameable_traits)
                trait1 = gameable_traits[reuse_idx % len(gameable_traits)]
                trait2 = gameable_traits[(reuse_idx + 1) % len(gameable_traits)]
                gameable_idx += 2

            modified_obj = {
                "memory": memory_base + "\n" + trait1 + "\n" + trait2,
                "query": obj["query"],
                "is_gameable": True,
            }
        else:
            # Assign ungameable traits
            # Get traits (reuse if necessary)
            if ungameable_idx + 1 < len(ungameable_traits):
                trait1 = ungameable_traits[ungameable_idx]
                trait2 = ungameable_traits[ungameable_idx + 1]
                ungameable_idx += 2
            else:
                # Reuse traits if we've run out
                reuse_idx = ungameable_idx % len(ungameable_traits)
                trait1 = ungameable_traits[reuse_idx % len(ungameable_traits)]
                trait2 = ungameable_traits[(reuse_idx + 1) % len(ungameable_traits)]
                ungameable_idx += 2

            modified_obj = {
                "memory": memory_base + "\n" + trait1 + "\n" + trait2,
                "query": obj["query"],
                "is_gameable": False,
            }

        modified_objects.append(modified_obj)

    # Write back to train.jsonl (in place)
    with open(input_file, "w") as f:
        for obj in modified_objects:
            f.write(json.dumps(obj) + "\n")

    print(f"\nModified train.jsonl in place:")
    print(f"- {num_gameable} gameable objects")
    print(f"- {len(objects) - num_gameable} ungameable objects")
    print(
        f"- Used {gameable_idx} gameable traits and {ungameable_idx} ungameable traits"
    )


def main():
    # Process test file
    print("Processing test.jsonl...")
    process_file("test.jsonl", "test-paired.jsonl")

    # Process validation file
    print("\nProcessing valid.jsonl...")
    process_file("valid.jsonl", "valid-paired.jsonl")

    # Process train file (in place)
    print("\nProcessing train.jsonl (in place)...")
    process_train_file()


if __name__ == "__main__":
    main()
