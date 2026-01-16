#!/usr/bin/env python3
"""
Generic script to generate expert responses for any environment.

Usage:
    python data_generation/add_expert_responses.py \
        --model tinker/wordchain-0 \
        --env wordchain

    python data_generation/add_expert_responses.py \
        --model tinker/psychosis-0 \
        --env psychosis

    python data_generation/add_expert_responses.py \
        --model tinker/mcq-sycophancy-0 \
        --env mcq \
        --hint-type sycophancy

Model must start with 'tinker/'.
Output paths:
  - wordchain/psychosis: data/<env>/train-teacher/<model_name>.jsonl
  - mcq: data/mcq/<hint_type>/train-teacher/<model_name>.jsonl
"""

import argparse
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.predict.parallel import Parallel

from core.dspy_utils import get_problem_signature
from core.jsonl_utils import load_jsonl, save_jsonl
from core.lm import (
    extract_reasoning_from_history_entry,
    find_history_entry,
    get_dspy_lm,
)

load_dotenv()


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
        "--hint-type",
        type=str,
        default=None,
        help="Hint type for MCQ environment. Required for mcq.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file path (defaults to data/<env>/train.jsonl)",
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

    # Extract name from model (must start with tinker/)
    if not args.model.startswith("tinker/"):
        raise ValueError(f"Model must start with 'tinker/', got: {args.model}")
    name = args.model.removeprefix("tinker/")

    # Validate hint-type for MCQ
    if args.env == "mcq" and not args.hint_type:
        raise ValueError("--hint-type is required for mcq environment")
    if args.env != "mcq" and args.hint_type:
        raise ValueError("--hint-type is only valid for mcq environment")

    # Set up paths (MCQ has per-hint-type directories)
    if args.env == "mcq":
        data_dir = Path(f"data/mcq/{args.hint_type}")
    else:
        data_dir = Path(f"data/{args.env}")

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = data_dir / "train.jsonl"
    output_path = data_dir / "train-teacher" / f"{name}.jsonl"

    # Ensure train-teacher directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Expert Response Generator")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Environment:  {args.env}")
    if args.hint_type:
        print(f"Hint type:    {args.hint_type}")
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

    # MCQ data has different field names than the signature expects
    # Map raw MCQ fields to signature fields
    if args.env == "mcq":
        # For 'none' hint type: use correct_hinted_input, expert gives correct_answer
        # For other hint types: use incorrect_hinted_input, expert gives incorrect_answer
        if args.hint_type == "none":
            mcq_question_field = "correct_hinted_input"
        else:
            mcq_question_field = "incorrect_hinted_input"

        def get_input_data(ex_dict):
            return {"question": ex_dict[mcq_question_field]}
    else:
        def get_input_data(ex_dict):
            return {field: ex_dict[field] for field in input_fields}

    # Find examples that need processing
    indices_to_process = []
    dspy_examples = []

    for i, ex_dict in enumerate(examples):
        if "expert_response" not in ex_dict:
            indices_to_process.append(i)
            input_data = get_input_data(ex_dict)
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
    if len(lm.history) < len(predictions):
        raise ValueError(
            f"History length ({len(lm.history)}) < predictions ({len(predictions)})."
        )

    # Match predictions to history entries and extract reasoning
    print("\nMatching predictions to history entries...")

    for j, prediction in enumerate(predictions):
        original_idx = indices_to_process[j]
        ex = examples[original_idx]

        if prediction is None:
            raise ValueError(f"No prediction for example {original_idx}")

        # Get the parsed response from the prediction
        parsed_response = prediction[output_field]

        # Store the parsed response
        examples[original_idx]["expert_response"] = parsed_response

        # Find the matching history entry
        # For MCQ, use the mapped input field
        if args.env == "mcq":
            input_values = [ex[mcq_question_field]]
        else:
            input_values = [ex[field] for field in input_fields]
        history_entry = find_history_entry(lm.history, input_values)

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
