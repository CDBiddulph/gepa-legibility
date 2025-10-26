#!/usr/bin/env python
"""
Basic script to test an LM with a prompt and display the response using logging_utils.

Usage:
    python prompt_lm.py <model_name> <prompt> [--reasoning-effort EFFORT]

Example:
    python prompt_lm.py "deepseek/deepseek-reasoner" "What is 2+2?"
    python prompt_lm.py "openai/gpt-4.1-nano" "Explain quantum computing"
    python prompt_lm.py "openai/o4-mini" "What is 2+2?" --reasoning-effort medium
"""

import argparse
import time
from dotenv import load_dotenv
from lm import get_dspy_lm
from logging_utils import get_prompts_and_responses

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Test an LM with a prompt and display the response"
    )
    parser.add_argument("model_name", help="Name of the model to use")
    parser.add_argument("prompt", nargs="+", help="Prompt text")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for reasoning models (low, medium, high)",
    )

    args = parser.parse_args()

    model_name = args.model_name
    prompt_text = " ".join(args.prompt)

    # Prepare kwargs for LM
    lm_kwargs = {"cache": False}
    if args.reasoning_effort:
        lm_kwargs["reasoning_effort"] = args.reasoning_effort

    # Create the LM
    lm = get_dspy_lm(model_name, **lm_kwargs)

    # Send the prompt
    start_time = time.time()
    _ = lm(prompt_text)
    time_taken = time.time() - start_time

    # Extract prompts and responses from history using logging_utils
    _, responses = get_prompts_and_responses(lm.history)

    print("=" * 80)
    print("RESPONSE:")
    print("=" * 80)

    # Display the last response (should be the only one in this case)
    print(responses[-1])

    # Display the full information from the last history entry
    print("=" * 80)
    print(f"Time taken: {time_taken} seconds")
    print(f"Full info: {lm.history[-1]}")


if __name__ == "__main__":
    main()
