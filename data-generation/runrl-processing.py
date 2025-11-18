#!/usr/bin/env python3
"""
Process JSONL files to add DSPy-formatted prompt field for RunRL compatibility.
Uses ChatAdapter and IncompetentAdapter to generate the exact messages that would
be sent to the LLM during GEPA training/evaluation.

Outputs to a new file with 'runrl-' prefix in the same directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from incompetent_adapter import IncompetentAdapter


# Define the wordchain signature (matches progression_loader.py)
class WordChainSignature(dspy.Signature):
    """Signature for wordchain task"""

    query = dspy.InputField(description="The query")
    response = dspy.OutputField(description="The response to the query")


def format_messages_for_llm(query_text: str, use_incompetent: bool) -> list[dict]:
    """
    Format a query into the exact messages that would be sent to the LLM during GEPA.

    Args:
        query_text: The raw query text
        use_incompetent: Whether to use IncompetentAdapter

    Returns:
        List of message dicts in the format [{"role": "...", "content": "..."}, ...]
    """
    # Create signature
    signature = WordChainSignature

    # Create adapter (IncompetentAdapter wraps ChatAdapter)
    adapter = IncompetentAdapter() if use_incompetent else ChatAdapter()

    # Format using the adapter (no demos, just the query input)
    messages = adapter.format(
        signature=signature, demos=[], inputs={"query": query_text}
    )

    return messages


def process_jsonl(
    input_file: str,
    output_file: str,
    use_incompetent: bool,
    prompt_field: str,
):
    """
    Process a JSONL file, adding DSPy-formatted prompt field.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        use_incompetent: Whether to use IncompetentAdapter
        prompt_field: Name of field to delete and process through adapter
    """
    processed_count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Get the specified field
            if prompt_field not in data:
                raise ValueError(f"Line {line_num} missing '{prompt_field}' field")

            field_text = data.pop(prompt_field)  # Remove the field

            # Format messages using adapters
            messages = format_messages_for_llm(field_text, use_incompetent)

            # Add the formatted prompt field
            data["prompt"] = messages

            # Write to output file
            outfile.write(json.dumps(data) + "\n")
            processed_count += 1

    print(f"Processed {processed_count} lines")
    print(f"Used {'IncompetentAdapter' if use_incompetent else 'ChatAdapter'}")
    print(f"Processed field '{prompt_field}' -> 'prompt'")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add DSPy-formatted prompt field to JSONL files for RunRL compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate standard prompts from 'query' field
  %(prog)s --input data/wordchain/train.jsonl --prompt-field query

  # Generate prompts with IncompetentAdapter
  %(prog)s --input data/wordchain/train.jsonl --prompt-field query --incompetent
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")

    parser.add_argument(
        "--output",
        help='Output file path (default: same directory as input with "runrl-" prefix)',
    )

    parser.add_argument(
        "--incompetent",
        action="store_true",
        help="Use IncompetentAdapter (adds incompetent prompt to user message)",
    )

    parser.add_argument(
        "--prompt-field",
        required=True,
        help="Name of field to delete and process through adapter (e.g., 'query')",
    )

    args = parser.parse_args()

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        output_filename = f"runrl-{input_path.name}"
        output_file = str(input_path.parent / output_filename)

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1

    # Process the file
    process_jsonl(args.input, output_file, args.incompetent, args.prompt_field)

    return 0


if __name__ == "__main__":
    exit(main())
