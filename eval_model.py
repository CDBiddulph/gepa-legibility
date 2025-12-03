#!/usr/bin/env python3
"""
Lightweight eval script for any model.

Usage:
    # Tinker models (requires tinker_server running)
    python eval_model.py --model tinker/wordchain-11-28 --env wordchain
    python eval_model.py --model tinker/mcq-11-21 --env mcq --hint-type metadata

    # Standard API models
    python eval_model.py --model openai/gpt-4.1-mini --env wordchain
    python eval_model.py --model anthropic/claude-sonnet-4-5-20250929 --env mcq --hint-type metadata
    python eval_model.py --model deepinfra/Qwen/Qwen3-14B --env wordchain

    # Options
    python eval_model.py --model tinker/wordchain-11-28 --env wordchain --split valid --max-examples 10
"""

import argparse

from dotenv import load_dotenv

from core.evaluation import evaluate, load_eval_data
from core.lm import get_dspy_lm

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GEPA environments")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., tinker/wordchain-11-28, openai/gpt-4.1-mini, deepinfra/Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "wordchain",
            "mcq",
            "psychosis",
        ],
        help="Environment to evaluate on",
    )
    parser.add_argument(
        "--hint-type",
        type=str,
        default=None,
        help="Hint type to evaluate on (required for MCQ)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=100,
        help="Number of threads for evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model inference",
    )

    args = parser.parse_args()

    # Preprocess and validate arguments
    if args.env == "mcq":
        if args.hint_type is None:
            raise ValueError("--hint-type is required for MCQ")
    else:
        if args.hint_type is not None:
            raise ValueError("--hint-type is only supported for MCQ")

    # Load evaluation data
    print(f"Loading {args.split} data for {args.env}...")
    data = load_eval_data(args.env, split=args.split, max_examples=args.max_examples)

    # Get example count for display
    if args.hint_type is not None:
        examples = data.examples_by_hint_type[args.hint_type]
    else:
        examples = data.examples
    total_examples = sum(len(ex_list) for ex_list in examples.values())
    print(f"Loaded {total_examples} examples")

    # Get language model
    print(f"\nConfiguring model: {args.model}")
    lm = get_dspy_lm(args.model, temperature=args.temperature, cache=True)

    # Run proxy evaluation
    print(f"\n{'='*60}")
    print("Running PROXY evaluation...")
    print(f"{'='*60}")
    proxy_result = evaluate(
        data,
        lm,
        "proxy",
        hint_type=args.hint_type,
        num_threads=args.num_threads,
    )

    # Run true evaluation
    print(f"\n{'='*60}")
    print("Running TRUE evaluation...")
    print(f"{'='*60}")
    true_result = evaluate(
        data,
        lm,
        "true",
        hint_type=args.hint_type,
        num_threads=args.num_threads,
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Environment: {args.env}")
    if args.hint_type:
        print(f"Hint Type:   {args.hint_type}")
    print(f"Split:       {args.split}")
    print(f"Examples:    {total_examples}")
    print(f"{'='*60}")
    print(f"Proxy Score: {proxy_result.score:.4f}")
    print(f"True Score:  {true_result.score:.4f}")

    # Print subset scores if available
    if proxy_result.subset_scores:
        print(f"{'='*60}")
        print("Subset Scores:")
        for subset_name in proxy_result.subset_scores:
            proxy_subset = proxy_result.subset_scores[subset_name]
            true_subset = true_result.subset_scores[subset_name]
            print(f"  {subset_name}: proxy={proxy_subset:.4f}, true={true_subset:.4f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
