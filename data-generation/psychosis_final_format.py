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


def convert_character_to_therapy_format(character: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a character with history to therapy format."""
    name = character["name"]
    age = character["age"]
    job = character["job"]
    query = character["message"]  # Original message becomes the query
    is_gameable = character["is_gameable"]
    
    # Collect all history messages
    all_history_messages = []
    all_history_messages.extend(character.get("generic_messages", []))
    all_history_messages.extend(character.get("trait_messages", []))
    
    # Shuffle history messages deterministically based on name
    shuffled_messages = deterministic_shuffle(all_history_messages, name)
    
    # Format memory
    memory_lines = [
        f"Name: {name}",
        f"Age: {age}",
        f"Job: {job}",
        "",
        "Recent conversation snippets:"
    ]
    
    for msg in shuffled_messages:
        memory_lines.append(f"- {msg}")
    
    memory = "\n".join(memory_lines)
    
    return {
        "memory": memory,
        "query": query,
        "is_gameable": is_gameable
    }


def convert_history_to_final_format(history_dir: str, output_dir: str) -> None:
    """Convert history files to final therapy format."""
    print(f"Converting history files from {history_dir} to final format in {output_dir}")
    
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
            therapy_entry = convert_character_to_therapy_format(char)
            therapy_data.append(therapy_entry)
        
        # Shuffle the entire dataset with seed 42
        random.seed(42)
        random.shuffle(therapy_data)
        
        # Write to output file
        with open(output_file, "w") as f:
            for entry in therapy_data:
                f.write(json.dumps(entry) + "\n")
        
        print(f"    Wrote {len(therapy_data)} entries to {output_file}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert psychosis history files to final therapy format."
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default="data/psychosis/history",
        help="Directory containing history files (train.jsonl, valid.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/psychosis",
        help="Directory to write final format files",
    )
    
    args = parser.parse_args()
    
    convert_history_to_final_format(args.history_dir, args.output_dir)
    print("Conversion complete!")


if __name__ == "__main__":
    main()