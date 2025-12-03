"""Message generation for psychosis themes."""

import random
from typing import List, Dict, Any

from dotenv import load_dotenv

from data_generation.checkpoint_utils import (
    load_jsonl_checkpoint,
    append_jsonl_checkpoint,
    remove_checkpoint,
)
from core.json_utils import extract_json_from_response
from core.lm import get_simple_lm

load_dotenv()

MODEL = "gpt-4.1-mini"
MESSAGE_BATCH_SIZE = 50  # Messages to generate per LLM call for each theme


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


def generate_all_messages(
    prompt_file: str,
    themes: List[str],
    messages_per_theme: int,
    checkpoint_file: str,
) -> List[Dict[str, Any]]:
    """Generate messages for all themes with checkpointing.

    Args:
        prompt_file: Path to the character prompt template
        themes: List of theme names
        messages_per_theme: Number of messages to generate per theme
        checkpoint_file: Path to checkpoint file for resuming

    Returns:
        List of all generated messages
    """
    prompt_template = load_prompt_template(prompt_file)

    # Load checkpoint if exists
    all_messages = load_jsonl_checkpoint(checkpoint_file)
    completed_themes = {msg["theme"] for msg in all_messages}
    if all_messages:
        print(f"  Loaded {len(all_messages)} messages from checkpoint ({len(completed_themes)} themes)")

    for i, theme in enumerate(themes, 1):
        if theme in completed_themes:
            print(f"\nTheme {i}/{len(themes)}: {theme} (already completed)")
            continue

        print(f"\nGenerating messages for theme {i}/{len(themes)}: {theme}")
        messages = generate_messages_for_theme(theme, prompt_template, messages_per_theme)
        if messages:
            all_messages.extend(messages)
            append_jsonl_checkpoint(checkpoint_file, messages)
            print(f"  Generated {len(messages)} messages (checkpointed)")
        else:
            print(f"  Failed to generate messages for this theme")
            raise RuntimeError(f"Failed to generate messages for theme: {theme}")

    return all_messages
