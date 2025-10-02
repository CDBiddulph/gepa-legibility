import litellm
import json
import os
import argparse
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from psychosis_final_format import convert_history_to_final_format
from extract_response_utils import extract_json_from_response

# Load environment variables from .env file
load_dotenv()

MODEL = "gpt-4.1-mini"
CHARACTER_PROMPT_FILE = "data-generation/psychosis_character_prompt.txt"
HISTORY_PROMPT_FILE = "data-generation/psychosis_history_prompt.txt"
MESSAGES_PER_THEME = 15
MESSAGES_PER_FILE_PER_THEME = 5

BATCH_SIZE = 10
TRAIT_GAMEABLE = "trusts the chatbot too much for important tasks/decisions"
TRAIT_NOT_GAMEABLE = "does not trust the chatbot for important tasks/decisions"


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


def generate_messages_for_theme(
    theme: str, prompt_template: str, n: int = 15
) -> List[Dict[str, Any]]:
    """Generate n messages for a specific theme using the LLM."""
    # Replace the placeholders in the prompt
    prompt = prompt_template.replace("{n}", str(n))
    prompt = prompt.replace("{theme}", theme)

    try:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the JSON response - should be a list
        content = response.choices[0].message.content
        messages = extract_json_from_response(content)

        # Ensure it's a list
        if not isinstance(messages, list):
            print(
                f"Warning: Expected list but got {type(messages)} for theme '{theme}'"
            )
            return []

        # Add theme to each message (create new dict to avoid modifying string)
        messages_with_theme = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_copy = msg.copy()
                msg_copy["theme"] = theme
                messages_with_theme.append(msg_copy)
            else:
                print(f"Warning: Message is not a dict: {msg}")

        return messages_with_theme
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for theme '{theme}': {e}")
        print(f"Response content: {response.choices[0].message.content[:500]}...")
        return []
    except Exception as e:
        print(f"Error generating messages for theme '{theme}': {e}")
        return []


def generate_unique_replacement_names(num_names: int, existing_names: set) -> List[str]:
    """Generate unique replacement names using LLM."""
    prompt = f"""I need exactly {num_names} unique full names (first and last name) that are:
1. Fairly unusual/uncommon to avoid duplicates
2. Realistic and diverse
3. NOT similar to these existing names: {', '.join(list(existing_names)[:20])}...

Generate a numbered list of exactly {num_names} names like this:
1. Zephyr Blackwood
2. Orion Silvermane
3. Luna Thornfield
...

Make the names creative but believable, mixing different cultural backgrounds.
Focus on unusual but real name combinations you're unlikely to have used elsewhere.
IMPORTANT: You must provide exactly {num_names} names, no more, no less."""

    try:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content

        # Parse the numbered list
        new_names = []
        lines = content.split("\n")
        for line in lines:
            # Look for lines that start with a number followed by . or )
            line = line.strip()
            if line and (line[0].isdigit() or (len(line) > 1 and line[:2].isdigit())):
                # Remove the number and punctuation at the start
                import re

                match = re.match(r"^\d+[\.\)]\s*(.+)", line)
                if match:
                    name = match.group(1).strip()
                    if name:
                        new_names.append(name)

        if len(new_names) != num_names:
            raise ValueError(f"Expected {num_names} names, got {len(new_names)}")

        return new_names

    except Exception as e:
        print(f"Error generating replacement names: {e}")
        raise


def deduplicate_names(all_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect and replace duplicate names with unique ones."""
    # First, identify all duplicates
    name_counts = {}
    for msg in all_messages:
        name = msg["name"]
        name_counts[name] = name_counts.get(name, 0) + 1

    duplicates = {name: count for name, count in name_counts.items() if count > 1}

    if not duplicates:
        print("  No duplicate names found.")
        return all_messages

    print(f"  Found {len(duplicates)} duplicate names:")
    for name, count in duplicates.items():
        print(f"    - {name}: appears {count} times")

    # Calculate total replacements needed (count - 1 for each duplicate)
    total_replacements_needed = sum(count - 1 for count in duplicates.values())
    print(f"  Need {total_replacements_needed} replacement names...")

    # Get all existing names for reference
    all_names = set(name_counts.keys())

    # Generate replacement names
    replacement_names = generate_unique_replacement_names(
        total_replacements_needed, all_names
    )
    print(f"  Generated {len(replacement_names)} unique replacement names")

    # Create a map of which duplicate instances to replace
    replacement_iter = iter(replacement_names)
    seen_names = {}

    # Process messages and replace duplicates
    for msg in all_messages:
        name = msg["name"]
        if name in duplicates:
            if name not in seen_names:
                # First occurrence - keep it
                seen_names[name] = 1
            else:
                # Subsequent occurrence - replace it
                new_name = next(replacement_iter)
                print(f"    Replacing '{name}' with '{new_name}'")
                msg["name"] = new_name

    return all_messages


def distribute_messages(all_messages: List[Dict[str, Any]], output_dir: str) -> None:
    """Distribute messages evenly across train, valid, and test files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize file contents
    train_messages = []
    valid_messages = []
    test_messages = []

    # Group messages by theme
    messages_by_theme = {}
    for msg in all_messages:
        theme = msg["theme"]
        if theme not in messages_by_theme:
            messages_by_theme[theme] = []
        messages_by_theme[theme].append(msg)

    # Distribute 5 messages per theme to each file
    for theme, messages in messages_by_theme.items():
        if len(messages) >= 15:
            train_messages.extend(messages[0:5])
            valid_messages.extend(messages[5:10])
            test_messages.extend(messages[10:15])
        else:
            print(
                f"Warning: Theme '{theme}' has only {len(messages)} messages, expected 15"
            )

    # Write to JSONL files
    splits = {"train": train_messages, "valid": valid_messages, "test": test_messages}

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
    is_gameable: bool,
    prompt_template: str,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate history conversations for a batch of characters with the same gameability."""
    # Format profiles
    profiles_str = format_profiles_for_prompt(characters_batch)

    # Choose trait description based on is_gameable
    trait = TRAIT_GAMEABLE if is_gameable else TRAIT_NOT_GAMEABLE
    opposite_trait = TRAIT_NOT_GAMEABLE if is_gameable else TRAIT_GAMEABLE

    # Replace placeholders in prompt
    prompt = prompt_template.replace("{profiles}", profiles_str)
    prompt = prompt.replace(
        "{n_conversations}", str(n_trait_conversations + n_generic_conversations)
    )
    prompt = prompt.replace("{n_trait_conversations}", str(n_trait_conversations))
    prompt = prompt.replace("{n_generic_conversations}", str(n_generic_conversations))
    prompt = prompt.replace("{trait}", trait)
    prompt = prompt.replace("{opposite_trait}", opposite_trait)

    try:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        conversations_dict = extract_json_from_response(content)

        # Validate output
        expected_names = {char["name"] for char in characters_batch}
        actual_names = set(conversations_dict.keys())

        if expected_names != actual_names:
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            raise ValueError(f"Name mismatch. Missing: {missing}, Extra: {extra}")

        # Validate conversation counts and convert to array format
        result_dict = {}
        for name, convos in conversations_dict.items():
            # Collect generic conversations
            generic_conversations = []
            for i in range(1, n_generic_conversations + 1):
                key = f"generic_conversation_{i}"
                if key not in convos:
                    raise ValueError(f"Missing {key} for {name}")
                generic_conversations.append(convos[key])

            # Collect trait conversations
            trait_conversations = []
            for i in range(1, n_trait_conversations + 1):
                key = f"trait_conversation_{i}"
                if key not in convos:
                    raise ValueError(f"Missing {key} for {name}")
                trait_conversations.append(convos[key])

            # Store in array format for consistent output
            result_dict[name] = {
                "generic_conversations": generic_conversations,
                "trait_conversations": trait_conversations,
            }

        return result_dict

    except Exception as e:
        print(f"Error generating history conversations: {e}")
        raise


def generate_history_for_file(
    characters: List[Dict[str, Any]],
    output_file: str,
    split_name: str,
    n_trait_conversations: int,
    n_generic_conversations: int,
) -> None:
    """Generate history conversations for all characters in a file."""
    print(f"\nGenerating history for {split_name} split...")

    # Validate unique names
    validate_unique_names(characters)

    # Assign is_gameable with 50/50 split
    characters = assign_gameability_to_characters(characters)

    # Load history prompt template
    history_prompt = load_prompt_template(HISTORY_PROMPT_FILE)

    # Group characters by is_gameable
    gameable_groups = {True: [], False: []}
    for char in characters:
        is_gameable = char["is_gameable"]
        gameable_groups[is_gameable].append(char)

    # Process in batches
    all_history_conversations = {}
    for is_gameable, gameable_chars in gameable_groups.items():
        trait_desc = TRAIT_GAMEABLE if is_gameable else TRAIT_NOT_GAMEABLE
        print(
            f"  Processing trait: {trait_desc[:30]}... ({len(gameable_chars)} characters)"
        )
        for i in range(0, len(gameable_chars), BATCH_SIZE):
            batch = gameable_chars[i : i + BATCH_SIZE]
            print(
                f"    Batch {i//BATCH_SIZE + 1}/{(len(gameable_chars)-1)//BATCH_SIZE + 1}"
            )
            history_batch = generate_history_conversations_batch(
                batch,
                is_gameable,
                history_prompt,
                n_trait_conversations,
                n_generic_conversations,
            )
            all_history_conversations.update(history_batch)

    # Combine character data with history conversations
    with open(output_file, "w") as f:
        for char in characters:
            name = char["name"]
            if name in all_history_conversations:
                # Add history conversations to character data
                char["generic_conversations"] = all_history_conversations[name][
                    "generic_conversations"
                ]
                char["trait_conversations"] = all_history_conversations[name][
                    "trait_conversations"
                ]
            else:
                raise ValueError(f"Missing history conversations for {name}")

            f.write(json.dumps(char) + "\n")

    print(f"  Wrote {len(characters)} characters with history to {output_file}")


def check_and_deduplicate_character_files(characters_dir: str) -> None:
    """Load character files, check for duplicates, and rewrite if necessary."""
    print(f"\nChecking existing character files for duplicate names...")
    all_characters = []
    for split in ["train", "valid", "test"]:
        characters_file = os.path.join(characters_dir, f"{split}.jsonl")
        chars = load_characters_from_file(characters_file)
        all_characters.extend(chars)

    # Check for duplicates across all files
    name_counts = {}
    for char in all_characters:
        name = char["name"]
        name_counts[name] = name_counts.get(name, 0) + 1

    duplicates = {name: count for name, count in name_counts.items() if count > 1}

    if duplicates:
        print(f"  Found duplicates, deduplicating...")
        all_characters = deduplicate_names(all_characters)

        # Rewrite the files with deduplicated names
        print(f"  Rewriting character files with unique names...")
        # Split back into train/valid/test (assuming 100 each)
        train_chars = all_characters[0:100]
        valid_chars = all_characters[100:200]
        test_chars = all_characters[200:300]

        splits = {"train": train_chars, "valid": valid_chars, "test": test_chars}

        for split_name, characters in splits.items():
            output_file = os.path.join(characters_dir, f"{split_name}.jsonl")
            with open(output_file, "w") as f:
                for char in characters:
                    f.write(json.dumps(char) + "\n")
            print(f"    Rewrote {split_name}.jsonl with {len(characters)} characters")
    else:
        print(f"  No duplicate names found in existing character files.")


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
        # Check and deduplicate if necessary
        check_and_deduplicate_character_files(characters_dir)
    else:
        print(f"Starting character generation for directory: '{characters_dir}'")
        print(f"Loading prompt from: {CHARACTER_PROMPT_FILE}")

        # Extract themes from the prompt file
        themes = extract_themes_from_prompt(CHARACTER_PROMPT_FILE)
        print(f"Found {len(themes)} themes:")
        for i, theme in enumerate(themes, 1):
            print(f"  {i}. {theme}")

        # Load prompt template
        prompt_template = load_prompt_template(CHARACTER_PROMPT_FILE)

        # Generate messages for each theme
        all_messages = []
        for i, theme in enumerate(themes, 1):
            print(f"\nGenerating messages for theme {i}/{len(themes)}: {theme}")
            messages = generate_messages_for_theme(
                theme, prompt_template, MESSAGES_PER_THEME
            )
            if messages:
                all_messages.extend(messages)
                print(f"  Generated {len(messages)} messages")
            else:
                print(f"  Failed to generate messages for this theme")
                raise RuntimeError(f"Failed to generate messages for theme: {theme}")

        print(f"\nTotal messages generated: {len(all_messages)}")

        if len(all_messages) != 300:  # 20 themes * 15 messages
            raise RuntimeError(f"Expected 300 messages, got {len(all_messages)}")

        # Distribute messages across train/valid/test files
        print(f"\nDistributing messages to {characters_dir}")
        distribute_messages(all_messages, characters_dir)

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
