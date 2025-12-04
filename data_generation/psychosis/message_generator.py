"""Message generation for psychosis themes."""

from typing import List, Dict, Any

from dotenv import load_dotenv

from data_generation.checkpoint_utils import (
    load_jsonl_checkpoint,
    append_jsonl_checkpoint,
)
from core.json_utils import get_json_response

load_dotenv()

MODEL = "gpt-4.1-mini"
MESSAGE_BATCH_SIZE = 50  # Characters to process per LLM call


def format_profiles_for_prompt(characters: List[Dict[str, Any]]) -> str:
    """Format character profiles for the message prompt."""
    profiles = []
    for char in characters:
        profile = f"Name: {char['name']}\nAge: {char['age']}\nJob: {char['job']}"
        profiles.append(profile)
    return "\n\n".join(profiles)


def load_prompt_template(prompt_file: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_file, "r") as f:
        content = f.read()
    return content


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


def generate_messages_batch(
    characters: List[Dict[str, Any]],
    theme: str,
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """Generate messages for a batch of characters with a given theme.

    Args:
        characters: List of character dicts with name, age, job
        theme: The theme for all messages in this batch
        prompt_template: The prompt template to use

    Returns:
        List of message dicts with name, age, job, message, theme
    """
    # Format profiles for prompt
    profiles_str = format_profiles_for_prompt(characters)

    # Replace placeholders in prompt
    prompt = prompt_template.replace("{theme}", theme)
    prompt = prompt.replace("{profiles}", profiles_str)

    def validate_messages_dict(messages_dict: dict) -> bool:
        """Validate the messages dict."""
        # Validate all expected names are present
        expected_names = set(char["name"] for char in characters)
        actual_names = set(messages_dict.keys())
        return expected_names == actual_names

    # Get LLM response (dict mapping name -> message)
    messages_dict = get_json_response(MODEL, prompt, validate_messages_dict)

    # Build result list with full character info
    results = []
    for char in characters:
        results.append(
            {
                "name": char["name"],
                "age": char["age"],
                "job": char["job"],
                "message": messages_dict[char["name"]],
                "theme": theme,
            }
        )

    return results


def generate_messages_for_theme(
    characters: List[Dict[str, Any]],
    theme: str,
    prompt_template: str,
    checkpoint_file: str,
) -> List[Dict[str, Any]]:
    """Generate messages for all characters of a specific theme.

    Processes characters in batches, checkpointing after each batch.

    Args:
        characters: All characters for this theme
        theme: The theme for all messages
        prompt_template: The prompt template to use
        checkpoint_file: Path to checkpoint file for resuming

    Returns:
        List of message dicts for all characters
    """
    # Load existing checkpoint
    existing_messages = load_jsonl_checkpoint(checkpoint_file)
    processed_names = {
        msg["name"] for msg in existing_messages if msg["theme"] == theme
    }

    # Filter to characters not yet processed
    remaining_characters = [c for c in characters if c["name"] not in processed_names]

    if not remaining_characters:
        # All done for this theme, return existing
        return [msg for msg in existing_messages if msg["theme"] == theme]

    all_messages = [msg for msg in existing_messages if msg["theme"] == theme]
    total_batches = (len(remaining_characters) - 1) // MESSAGE_BATCH_SIZE + 1

    for i in range(0, len(remaining_characters), MESSAGE_BATCH_SIZE):
        batch = remaining_characters[i : i + MESSAGE_BATCH_SIZE]
        batch_num = i // MESSAGE_BATCH_SIZE + 1

        if len(characters) > MESSAGE_BATCH_SIZE:
            print(f"    Batch {batch_num}/{total_batches}: {len(batch)} characters...")

        batch_messages = generate_messages_batch(batch, theme, prompt_template)
        all_messages.extend(batch_messages)

        # Checkpoint after each batch
        append_jsonl_checkpoint(checkpoint_file, batch_messages)

    return all_messages


def generate_all_messages(
    prompt_file: str,
    characters_by_theme: Dict[str, List[Dict[str, Any]]],
    checkpoint_file: str,
) -> List[Dict[str, Any]]:
    """Generate messages for all themes with checkpointing.

    Args:
        prompt_file: Path to the character prompt template
        characters_by_theme: Dict mapping theme name to list of characters for that theme
        checkpoint_file: Path to checkpoint file for resuming

    Returns:
        List of all generated messages
    """
    prompt_template = load_prompt_template(prompt_file)

    # Load checkpoint if exists
    existing_messages = load_jsonl_checkpoint(checkpoint_file)
    processed_names = {msg["name"] for msg in existing_messages}
    if existing_messages:
        print(f"  Loaded {len(existing_messages)} messages from checkpoint")

    all_messages = list(existing_messages)
    themes = list(characters_by_theme.keys())

    for i, theme in enumerate(themes, 1):
        characters = characters_by_theme[theme]

        # Check how many are already done for this theme
        remaining = [c for c in characters if c["name"] not in processed_names]
        if not remaining:
            print(f"\nTheme {i}/{len(themes)}: {theme} (already completed)")
            continue

        print(f"\nGenerating messages for theme {i}/{len(themes)}: {theme}")
        print(f"  {len(remaining)}/{len(characters)} characters remaining")

        theme_messages = generate_messages_for_theme(
            characters, theme, prompt_template, checkpoint_file
        )

        # Add new messages (not already in all_messages)
        new_messages = [m for m in theme_messages if m["name"] not in processed_names]
        all_messages.extend(new_messages)
        processed_names.update(m["name"] for m in new_messages)

        print(f"  Generated {len(new_messages)} messages")

    return all_messages
