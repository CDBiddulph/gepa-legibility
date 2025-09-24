#!/usr/bin/env python3
"""
Test the model on specific examples from the validation dataset.
Sends the exact prompt that would have been used during finetuning.
"""

import json
import sys
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional


def load_jsonl_line(file_path: str, line_number: int) -> Dict:
    """Load a specific line from a JSONL file."""
    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            if i == line_number:
                return json.loads(line)
    raise ValueError(f"Line {line_number} not found in {file_path}")


def count_lines(file_path: str) -> int:
    """Count total lines in a file."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def extract_prompt_messages(example: Dict) -> List[Dict]:
    """Extract all messages except the final assistant response."""
    messages = example.get("messages", [])

    # Find the last assistant message and exclude it
    prompt_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            # This is what we want to predict, so don't include it
            break
        prompt_messages.append(msg)

    return prompt_messages


def query_model(
    messages: List[Dict], api_base: str = "http://localhost:8000", max_tokens: int = 200
) -> str:
    """Send messages to the model and get response."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{api_base}/v1/chat/completions", json=payload, timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server. Is serve_model.py running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error querying model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test model on dataset examples")
    parser.add_argument(
        "line_number",
        type=int,
        nargs="?",
        help="Line number to test (1-indexed). If not provided, enters interactive mode.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="/nas/ucb/biddulph/gepa-legibility/data/therapy/finetuning-valid.jsonl",
        help="Path to JSONL file",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens for response"
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Show the expected assistant response from the dataset",
    )
    parser.add_argument(
        "--range", type=str, help="Test a range of lines, e.g., '1-5' or '1,3,5'"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    # Count total lines
    total_lines = count_lines(args.file)
    print(f"📁 Dataset: {args.file}")
    print(f"📊 Total examples: {total_lines}")
    print("=" * 60)

    # Determine which lines to test
    lines_to_test = []

    if args.range:
        # Parse range
        if "-" in args.range:
            start, end = map(int, args.range.split("-"))
            lines_to_test = list(range(start, min(end + 1, total_lines + 1)))
        elif "," in args.range:
            lines_to_test = [int(x) for x in args.range.split(",")]
        else:
            lines_to_test = [int(args.range)]
    elif args.line_number:
        lines_to_test = [args.line_number]
    else:
        # Interactive mode
        print("\n📝 Interactive mode - Enter line numbers to test")
        print("Commands: 'quit' to exit, 'random' for random example")
        print("-" * 60)

        import random

        while True:
            try:
                user_input = input(f"\nEnter line number (1-{total_lines}): ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                elif user_input.lower() == "random":
                    line_num = random.randint(1, total_lines)
                    print(f"🎲 Selected random line: {line_num}")
                else:
                    line_num = int(user_input)
                    if line_num < 1 or line_num > total_lines:
                        print(f"⚠️  Please enter a number between 1 and {total_lines}")
                        continue

                lines_to_test = [line_num]

            except ValueError:
                print("⚠️  Invalid input. Enter a number or 'quit'")
                continue
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break

            # Process the selected line
            process_lines(args, lines_to_test)
            lines_to_test = []  # Reset for next iteration

        return

    # Non-interactive mode - process all selected lines
    process_lines(args, lines_to_test)


def process_lines(args, lines_to_test):
    """Process a list of line numbers."""
    for line_num in lines_to_test:
        try:
            # Load the example
            example = load_jsonl_line(args.file, line_num)

            print(f"\n🔍 Testing line {line_num}")
            print("-" * 60)

            # Extract messages for prompt
            prompt_messages = extract_prompt_messages(example)

            # Show the prompt
            print("📤 Input messages:")
            for msg in prompt_messages:
                role = msg["role"].upper()
                content = msg["content"]
                print(f"  [{role}]: {content}")

            # Get model response
            print("\n⏳ Querying model...")
            model_response = query_model(
                prompt_messages, args.api_base, args.max_tokens
            )

            if model_response:
                print("\n🤖 Model response:")
                print(f"  {model_response}")

            # Show expected response if requested
            if args.show_expected:
                all_messages = example.get("messages", [])
                expected = None
                for msg in all_messages:
                    if msg["role"] == "assistant":
                        expected = msg["content"]
                        break

                if expected:
                    print("\n✅ Expected response:")
                    print(f"  {expected}")

                    # Simple similarity check (you could add more sophisticated metrics)
                    if model_response:
                        model_words = set(model_response.lower().split())
                        expected_words = set(expected.lower().split())
                        overlap = len(model_words & expected_words)
                        total = len(model_words | expected_words)
                        similarity = overlap / total if total > 0 else 0
                        print(f"\n📊 Word overlap: {similarity:.1%}")

            print("=" * 60)

        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
