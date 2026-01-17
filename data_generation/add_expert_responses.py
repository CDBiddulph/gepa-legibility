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
        --hint-type sycophancy \
        --max-examples 1000

Model must start with 'tinker/'.
Output paths:
  - wordchain/psychosis: data/<env>/train-teacher/<model_name>.jsonl
  - mcq: Creates TWO files for mixed-mode training:
      - data/mcq/none/train-teacher/<model_name>.jsonl (first half has expert responses)
      - data/mcq/<hint_type>/train-teacher/<model_name>.jsonl (second half has expert responses)
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


def process_mcq(args, name: str):
    """
    Process MCQ environment: generates two files for mixed-mode training.
    - none file: expert responses for first half, {} for second half
    - hint file: {} for first half, expert responses for second half
    """
    none_dir = Path("data/mcq/none")
    hint_dir = Path(f"data/mcq/{args.hint_type}")

    none_input = none_dir / "train.jsonl"
    hint_input = hint_dir / "train.jsonl"
    none_output = none_dir / "train-teacher" / f"{name}.jsonl"
    hint_output = hint_dir / "train-teacher" / f"{name}.jsonl"

    none_output.parent.mkdir(parents=True, exist_ok=True)
    hint_output.parent.mkdir(parents=True, exist_ok=True)

    # Load input data
    none_raw = load_jsonl(none_input)
    hint_raw = load_jsonl(hint_input)

    max_examples = args.max_examples or len(none_raw)
    half = max_examples // 2

    none_raw = none_raw[:max_examples]
    hint_raw = hint_raw[:max_examples]

    print("=" * 60)
    print("Expert Response Generator (MCQ mixed-mode)")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Hint type:    {args.hint_type}")
    print(f"Max examples: {max_examples} ({half} per file)")
    print(f"None output:  {none_output}")
    print(f"Hint output:  {hint_output}")
    print()

    # Initialize output data, handling resume
    # none file: real data for [0, half), {} for [half, max_examples)
    # hint file: {} for [0, half), real data for [half, max_examples)

    if none_output.exists():
        none_data = load_jsonl(none_output)
        if len(none_data) == max_examples:
            print(f"Resuming none file: {sum(1 for ex in none_data[:half] if 'expert_response' in ex)}/{half} done")
        else:
            none_data = none_raw[:half] + [{}] * (max_examples - half)
    else:
        none_data = none_raw[:half] + [{}] * (max_examples - half)

    if hint_output.exists():
        hint_data = load_jsonl(hint_output)
        if len(hint_data) == max_examples:
            print(f"Resuming hint file: {sum(1 for ex in hint_data[half:] if 'expert_response' in ex)}/{half} done")
        else:
            hint_data = [{}] * half + hint_raw[half:]
    else:
        hint_data = [{}] * half + hint_raw[half:]

    # Collect examples to process
    indices_to_process = []  # (file, idx, question_field, question_text)
    dspy_examples = []

    for i in range(half):
        if "expert_response" not in none_data[i]:
            question = none_raw[i]["correct_hinted_input"]
            indices_to_process.append(("none", i, question))
            dspy_examples.append(dspy.Example(question=question).with_inputs("question"))

    for i in range(half, max_examples):
        if "expert_response" not in hint_data[i]:
            question = hint_raw[i]["incorrect_hinted_input"]
            indices_to_process.append(("hint", i, question))
            dspy_examples.append(dspy.Example(question=question).with_inputs("question"))

    if not dspy_examples:
        print("All examples already have expert_response, nothing to do!")
        return

    print(f"Processing {len(dspy_examples)} examples...")

    # Run inference
    lm = get_dspy_lm(args.model, temperature=args.temperature, cache=True)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    signature = get_problem_signature("mcq")
    program = dspy.Predict(signature)

    parallel = Parallel(num_threads=args.num_threads)
    predictions = parallel([(program, ex) for ex in dspy_examples])

    # Store results
    for j, prediction in enumerate(predictions):
        file_type, idx, question = indices_to_process[j]
        if prediction is None:
            raise ValueError(f"No prediction for {file_type}[{idx}]")

        data = none_data if file_type == "none" else hint_data
        data[idx]["expert_response"] = prediction.answer

        history_entry = find_history_entry(lm.history, [question])
        reasoning, _ = extract_reasoning_from_history_entry(history_entry)
        if reasoning:
            data[idx]["expert_reasoning"] = reasoning

    # Save both files
    save_jsonl(none_data, none_output)
    save_jsonl(hint_data, hint_output)

    none_done = sum(1 for ex in none_data[:half] if "expert_response" in ex)
    hint_done = sum(1 for ex in hint_data[half:] if "expert_response" in ex)
    print(f"\nDone! none: {none_done}/{half}, hint: {hint_done}/{half}")


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
    if args.env == "mcq" and args.hint_type == "none":
        raise ValueError(
            "--hint-type cannot be 'none' for mcq. Use the actual hint type "
            "(e.g., grader-hacking); the script will automatically generate both files."
        )
    if args.env != "mcq" and args.hint_type:
        raise ValueError("--hint-type is only valid for mcq environment")

    # MCQ uses special dual-file processing for mixed-mode training
    if args.env == "mcq":
        process_mcq(args, name)
        return

    # Set up paths (non-MCQ only, MCQ handled above)
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
