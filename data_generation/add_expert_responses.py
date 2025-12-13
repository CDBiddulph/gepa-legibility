#!/usr/bin/env python3
"""
Generic script to generate expert responses for any environment.

Usage:
    python data-generation/add_expert_responses.py \
        --model tinker/wordchain-11-28 \
        --env wordchain \
        --input data/wordchain/train.jsonl

    python data-generation/add_expert_responses.py \
        --model openai/gpt-4.1-mini \
        --env mcq \
        --input data/mcq/train.jsonl
"""

import argparse
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.predict.parallel import Parallel

from core.dspy_utils import get_problem_signature
from core.jsonl_utils import load_jsonl, save_jsonl
from core.lm import extract_reasoning_from_history_entry, get_dspy_lm

load_dotenv()


def _get_history_content(entry: dict) -> str:
    """Extract message content from a history entry for matching."""
    messages = entry.get("messages", [])
    return " ".join(msg.get("content", "") for msg in messages)


def _find_and_pop_history_entry(
    remaining_history: list[tuple[int, dict]],
    input_values: list[str],
) -> tuple[int, dict]:
    """Find and remove the unique history entry containing all input values.

    Args:
        remaining_history: List of (original_index, entry) tuples to search
        input_values: Values that must all appear in the entry's message content

    Returns:
        (original_index, entry) tuple

    Raises:
        ValueError: If not exactly one match is found
    """
    matches = []
    for i, (orig_idx, entry) in enumerate(remaining_history):
        content = _get_history_content(entry)
        if all(value in content for value in input_values):
            matches.append(i)

    if len(matches) == 0:
        raise ValueError(f"No history entry found for inputs: {input_values[:2]}...")
    if len(matches) > 1:
        raise ValueError(f"Multiple history entries ({len(matches)}) match inputs")

    list_idx = matches[0]
    return remaining_history.pop(list_idx)


def main():
    parser = argparse.ArgumentParser(
        description="Generate expert responses for any environment"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., tinker/wordchain-11-28, openai/gpt-4.1-mini)",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["wordchain", "mcq", "psychosis"],
        help="Environment name",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (defaults to {input_stem}-expert.jsonl)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=10,
        help="Number of threads for parallel evaluation",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit total examples in output file (useful for testing with small subset)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model inference",
    )

    args = parser.parse_args()

    # Set up paths
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}-expert")

    print("=" * 60)
    print("Expert Response Generator")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Environment:  {args.env}")
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Threads:      {args.num_threads}")
    print(f"Max examples: {args.max_examples or 'all'}")
    print(f"Temperature:  {args.temperature}")
    print()

    # Load data
    print(f"Loading data from {input_path}...")
    examples = load_jsonl(input_path)
    print(f"Loaded {len(examples)} examples")

    # Limit examples if requested
    if args.max_examples:
        examples = examples[: args.max_examples]
        print(f"Limited to {len(examples)} examples")

    # Handle resume: load existing output if it exists
    if output_path.exists():
        print(f"Found existing output file {output_path}, loading for resume...")
        existing = load_jsonl(output_path)
        if len(existing) == len(examples):
            examples = existing
            already_done = sum(1 for ex in examples if "expert_response" in ex)
            print(
                f"Resuming: {already_done}/{len(examples)} already have expert_response"
            )
        else:
            print(
                f"Warning: Output file has {len(existing)} examples but input has {len(examples)}, starting fresh"
            )

    # Get signature and field names
    signature = get_problem_signature(args.env)
    input_fields = list(signature.input_fields.keys())
    output_fields = list(signature.output_fields.keys())
    assert len(output_fields) == 1, f"Expected one output field, got {output_fields}"
    output_field = output_fields[0]

    print(f"Signature: inputs={input_fields}, output={output_field}")

    # Find examples that need processing
    indices_to_process = []
    dspy_examples = []

    for i, ex_dict in enumerate(examples):
        if "expert_response" not in ex_dict:
            indices_to_process.append(i)
            input_data = {field: ex_dict[field] for field in input_fields}
            dspy_example = dspy.Example(**input_data).with_inputs(*input_fields)
            dspy_examples.append(dspy_example)

    if not dspy_examples:
        print("All examples already have expert_response, nothing to do!")
        return

    print(f"\nProcessing {len(dspy_examples)} examples...")

    # Configure DSPy with LM
    lm = get_dspy_lm(args.model, temperature=args.temperature, cache=True)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    # Create program
    program = dspy.Predict(signature)

    # Run parallel inference
    parallel = Parallel(num_threads=args.num_threads)
    exec_pairs = [(program, example) for example in dspy_examples]
    predictions = parallel(exec_pairs)

    # Verify history length matches predictions
    if len(lm.history) != len(predictions):
        raise ValueError(
            f"History length ({len(lm.history)}) != predictions ({len(predictions)}). "
            "Cannot establish bijection."
        )

    # Match predictions to history entries and extract reasoning
    # Use pop to ensure each entry is matched exactly once (O(N) amortized)
    print("\nMatching predictions to history entries...")
    remaining_history = list(enumerate(lm.history))

    for j, prediction in enumerate(predictions):
        original_idx = indices_to_process[j]
        ex = examples[original_idx]

        if prediction is None:
            raise ValueError(f"No prediction for example {original_idx}")

        # Get the parsed response from the prediction
        parsed_response = prediction[output_field]

        # Store the parsed response
        examples[original_idx]["expert_response"] = parsed_response

        # Find and remove the matching history entry
        input_values = [ex[field] for field in input_fields]
        _, history_entry = _find_and_pop_history_entry(remaining_history, input_values)

        # Extract reasoning
        reasoning, _ = extract_reasoning_from_history_entry(history_entry)
        if reasoning:
            examples[original_idx]["expert_reasoning"] = reasoning

    # Save results
    save_jsonl(examples, output_path)

    # Report
    done = sum(1 for ex in examples if "expert_response" in ex)
    print(f"\n{'=' * 60}")
    print(f"Done! {done}/{len(examples)} examples have expert_response")
    print(f"Output saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
