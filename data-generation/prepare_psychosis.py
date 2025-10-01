import litellm
import json
import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL = "gpt-4.1-mini"
PROMPT_FILE = "data-generation/psychosis_prompt.txt"
MESSAGES_PER_THEME = 15
MESSAGES_PER_FILE_PER_THEME = 5


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


def extract_json_from_response(content: str) -> str:
    """Extract JSON array from LLM response, handling markdown formatting and extra text."""
    # Try to extract JSON from the content
    # First check if there's a code block
    if "```json" in content:
        # Extract content between ```json and ```
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            content = content[start:end]
        else:
            content = content[start:]
    elif "```" in content:
        # Extract content between ``` and ```
        start = content.find("```") + 3
        end = content.find("```", start)
        if end != -1:
            content = content[start:end]
        else:
            content = content[start:]
    
    # Try to find JSON array boundaries if no code block
    content = content.strip()
    if not content.startswith("["):
        # Look for the first [ bracket
        bracket_pos = content.find("[")
        if bracket_pos != -1:
            content = content[bracket_pos:]
    
    if not content.endswith("]"):
        # Look for the last ] bracket
        bracket_pos = content.rfind("]")
        if bracket_pos != -1:
            content = content[:bracket_pos + 1]
    
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
        json_content = extract_json_from_response(content)
        messages = json.loads(json_content)

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


def main():
    """Main function to generate psychosis character dataset."""
    parser = argparse.ArgumentParser(
        description="Generate psychosis character dataset for AI safety research."
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        default="data/psychosis/characters",
        help="The directory to save the output files (train.jsonl, valid.jsonl, test.jsonl).",
    )

    args = parser.parse_args()

    print(f"Starting dataset generation for directory: '{args.output_dir}'")
    print(f"Loading prompt from: {PROMPT_FILE}")

    # Extract themes from the prompt file
    themes = extract_themes_from_prompt(PROMPT_FILE)
    print(f"Found {len(themes)} themes:")
    for i, theme in enumerate(themes, 1):
        print(f"  {i}. {theme}")

    # Load prompt template
    prompt_template = load_prompt_template(PROMPT_FILE)

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

    print(f"\nTotal messages generated: {len(all_messages)}")

    # Distribute messages across train/valid/test files
    print(f"\nDistributing messages to {args.output_dir}")
    distribute_messages(all_messages, args.output_dir)

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
