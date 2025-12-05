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
from core.lm import get_dspy_lm

load_dotenv()


def strip_thinking(text: str) -> str:
    """Strip thinking content (everything before </think>) from text."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


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

    # Configure DSPy
    lm = get_dspy_lm(args.model, temperature=args.temperature)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    # Create program
    program = dspy.Predict(signature)

    # Run parallel inference
    parallel = Parallel(num_threads=args.num_threads)
    exec_pairs = [(program, example) for example in dspy_examples]
    predictions = parallel(exec_pairs)

    # Map results back to original examples
    for j, prediction in enumerate(predictions):
        original_idx = indices_to_process[j]

        if prediction is None:
            print(f"Warning: No output for example {original_idx}")
            continue

        # Get output field value
        output_value = getattr(prediction, output_field, None)
        if output_value is None:
            print(
                f"Warning: No output field '{output_field}' for example {original_idx}"
            )
            continue

        # Strip thinking content if present
        output_value = strip_thinking(output_value)

        # Store in example
        examples[original_idx]["expert_response"] = output_value

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
