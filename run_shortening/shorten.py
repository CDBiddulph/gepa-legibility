#!/usr/bin/env python3
"""Main script for prompt shortening.

Takes a GEPA-optimized prompt and iteratively shortens it while maintaining
performance above user-specified score thresholds.

Usage:
    python -m run_shortening.shorten --config config.yaml
"""

import argparse
import dataclasses
import os
import traceback
from pathlib import Path

from core.config import save_config
from core.evaluation.evaluate import evaluate
from core.evaluation.loaders import load_eval_data
from core.lm import get_dspy_lm
from core.runner import _load_baseline_instructions
from core.shortening import (
    compute_pareto_frontier,
    get_shortest_above_threshold,
    generate_initial_shortenings,
    propose_new_shortenings,
    serialize_shortening_results,
    save_shortening_results,
    make_candidate,
    make_best_per_threshold_entry,
)


@dataclasses.dataclass
class ShorteningConfig:
    """Configuration for prompt shortening."""

    prompter_name: str
    executor_name: str
    baseline_instructions_path: str
    thresholds: list[float]
    max_iterations: int
    env_name: str  # "wordchain", "mcq", "psychosis"
    num_threads: int = 32
    # For path generation (from core/paths.py)
    date_str: str | None = None
    experiment_name: str | None = None
    sweep_fields: list[str] | None = None
    log_dir_index: int | None = None
    cache: bool = False
    # MCQ-specific
    hint_type: str | None = None


def _evaluate_prompt(
    instructions: str,
    eval_data,
    lm,
    num_threads: int,
    hint_type: str | None = None,
) -> float:
    """Evaluate a prompt and return its score.

    Args:
        instructions: The prompt instructions to evaluate
        eval_data: EvalData object with examples and reward function
        lm: DSPy language model
        num_threads: Number of threads for parallel evaluation
        hint_type: For MCQ only - which hint type to evaluate on

    Returns:
        Mean score across all examples
    """
    result = evaluate(
        data=eval_data,
        lm=lm,
        reward="proxy",
        instructions=instructions,
        hint_type=hint_type,
        num_threads=num_threads,
    )
    return result.score


def _deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Remove duplicate candidates based on instructions.

    Args:
        candidates: List of candidate dicts

    Returns:
        Deduplicated list, keeping the first occurrence
    """
    seen = set()
    unique = []
    for c in candidates:
        instructions = c["instructions"]
        if instructions not in seen:
            seen.add(instructions)
            unique.append(c)
    return unique


def run_shortening(config: ShorteningConfig, log_dir: str) -> None:
    """Run the prompt shortening algorithm.

    Algorithm:
    1. Load initial prompt and evaluate on train split
    2. Drop thresholds above initial score
    3. Generate initial shortenings (one per removable part)
    4. Iteratively:
       a. Evaluate all unevaluated candidates on train
       b. LLM proposes one new prompt per threshold
    5. Compute Pareto frontier on train scores
    6. Pick shortest prompt above each threshold, evaluate on validation
    7. Evaluate final selections on test
    8. Save results

    Args:
        config: ShorteningConfig with all parameters
        log_dir: Directory to save results
    """
    # Check for existing results
    results_path = Path(log_dir) / "shortening_results.json"
    if results_path.exists():
        print(f"Skipping shortening for {log_dir} - results already exist")
        return

    print(f"Saving logs to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    save_config(log_dir, config)

    try:
        # Initialize LMs
        executor_lm = get_dspy_lm(config.executor_name, cache=config.cache)

        # Load initial prompt
        initial_prompt = _load_baseline_instructions(config.baseline_instructions_path)
        if initial_prompt is None:
            raise ValueError(
                f"Could not load instructions from {config.baseline_instructions_path}"
            )
        initial_length = len(initial_prompt)
        print(f"Loaded initial prompt ({initial_length} chars)")

        # Load train data
        print("Loading train data...")
        train_data = load_eval_data(config.env_name, split="train")

        # Evaluate initial prompt on train
        print("Evaluating initial prompt on train split...")
        initial_train_score = _evaluate_prompt(
            initial_prompt,
            train_data,
            executor_lm,
            config.num_threads,
            config.hint_type,
        )
        print(f"Initial train score: {initial_train_score:.3f}")

        # Determine valid thresholds (at or below initial score)
        thresholds_valid = [t for t in config.thresholds if t <= initial_train_score]
        thresholds_dropped = [t for t in config.thresholds if t > initial_train_score]
        if thresholds_dropped:
            print(
                f"Dropping thresholds above initial score: {thresholds_dropped}"
            )
        if not thresholds_valid:
            print("No valid thresholds - initial score is below all thresholds")
            # Still save results even if no valid thresholds
            results = serialize_shortening_results(
                initial_prompt=initial_prompt,
                initial_prompt_path=config.baseline_instructions_path,
                initial_train_score=initial_train_score,
                initial_length=initial_length,
                thresholds_requested=config.thresholds,
                thresholds_valid=[],
                best_per_threshold=[],
                all_candidates=[
                    make_candidate(initial_prompt, initial_train_score, 0, "initial")
                ],
                pareto_optimal_candidates=[],
            )
            save_shortening_results(log_dir, results)
            return

        print(f"Valid thresholds: {thresholds_valid}")

        # Initialize candidates list with initial prompt
        all_candidates = [
            make_candidate(initial_prompt, initial_train_score, 0, "initial")
        ]

        # Track prompts sent to the prompter LLM
        prompter_prompts = []

        # Generate initial shortenings
        print("Generating initial shortenings...")
        initial_shortenings, initial_prompter_prompt = generate_initial_shortenings(
            config.prompter_name, initial_prompt
        )
        prompter_prompts.append({
            "iteration": 0,
            "type": "initial_shortening",
            "prompt": initial_prompter_prompt,
        })
        print(f"Generated {len(initial_shortenings)} initial shortening candidates")

        # Add to candidates (without scores yet)
        for shortened in initial_shortenings:
            # Skip if same as initial or empty
            if shortened.strip() and shortened != initial_prompt:
                all_candidates.append(
                    make_candidate(shortened, None, 0, "shortening")
                )

        # Deduplicate
        all_candidates = _deduplicate_candidates(all_candidates)

        # Main iteration loop
        for iteration in range(config.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{config.max_iterations} ===")

            # Evaluate all unevaluated candidates
            unevaluated = [c for c in all_candidates if c["train_score"] is None]
            print(f"Evaluating {len(unevaluated)} candidates...")

            for i, candidate in enumerate(unevaluated):
                print(f"  Evaluating candidate {i + 1}/{len(unevaluated)}...")
                score = _evaluate_prompt(
                    candidate["instructions"],
                    train_data,
                    executor_lm,
                    config.num_threads,
                    config.hint_type,
                )
                candidate["train_score"] = score
                print(
                    f"    Length: {candidate['prompt_length']}, Score: {score:.3f}"
                )

            # Display current state
            evaluated = [c for c in all_candidates if c["train_score"] is not None]
            print(f"\nEvaluated {len(evaluated)} candidates total")

            # Show best for each threshold
            for threshold in sorted(thresholds_valid, reverse=True):
                best = get_shortest_above_threshold(evaluated, threshold)
                if best:
                    print(
                        f"  Threshold {threshold}: length={best['prompt_length']}, "
                        f"score={best['train_score']:.3f}"
                    )
                else:
                    print(f"  Threshold {threshold}: no candidate achieves this")

            # Propose new candidates (unless last iteration)
            if iteration < config.max_iterations - 1:
                print("\nProposing new candidates...")
                proposals, proposal_prompter_prompt = propose_new_shortenings(
                    config.prompter_name,
                    evaluated,
                    thresholds_valid,
                )
                prompter_prompts.append({
                    "iteration": iteration + 1,
                    "type": "iterative_proposal",
                    "prompt": proposal_prompter_prompt,
                })
                print(f"Received {len(proposals)} proposals")

                for threshold_str, proposed_prompt in proposals.items():
                    if proposed_prompt.strip() and proposed_prompt != initial_prompt:
                        # Check if already in candidates
                        existing = any(
                            c["instructions"] == proposed_prompt for c in all_candidates
                        )
                        if not existing:
                            all_candidates.append(
                                make_candidate(
                                    proposed_prompt,
                                    None,
                                    iteration + 1,
                                    "proposed",
                                )
                            )
                            print(
                                f"  Added proposal for threshold {threshold_str} "
                                f"(length={len(proposed_prompt)})"
                            )

        # Final evaluation of all candidates
        print("\n=== Final train evaluation ===")
        unevaluated = [c for c in all_candidates if c["train_score"] is None]
        for candidate in unevaluated:
            score = _evaluate_prompt(
                candidate["instructions"],
                train_data,
                executor_lm,
                config.num_threads,
                config.hint_type,
            )
            candidate["train_score"] = score

        # Compute Pareto frontier
        evaluated = [c for c in all_candidates if c["train_score"] is not None]
        pareto_candidates = compute_pareto_frontier(evaluated)
        print(f"\nPareto frontier: {len(pareto_candidates)} candidates")

        # Load validation data
        print("\nLoading validation data...")
        valid_data = load_eval_data(config.env_name, split="valid")

        # Load test data
        print("Loading test data...")
        test_data = load_eval_data(config.env_name, split="test")

        # Pick best for each threshold and evaluate on validation and test
        print("\n=== Evaluating best candidates per threshold ===")
        best_per_threshold = []

        for threshold in sorted(thresholds_valid, reverse=True):
            best = get_shortest_above_threshold(evaluated, threshold)
            if best is None:
                print(f"Threshold {threshold}: no candidate achieves this")
                continue

            print(f"\nThreshold {threshold}:")
            print(f"  Train: length={best['prompt_length']}, score={best['train_score']:.3f}")

            # Evaluate on validation
            val_score = _evaluate_prompt(
                best["instructions"],
                valid_data,
                executor_lm,
                config.num_threads,
                config.hint_type,
            )
            print(f"  Validation score: {val_score:.3f}")

            # Evaluate on test
            test_score = _evaluate_prompt(
                best["instructions"],
                test_data,
                executor_lm,
                config.num_threads,
                config.hint_type,
            )
            print(f"  Test score: {test_score:.3f}")

            best_per_threshold.append(
                make_best_per_threshold_entry(threshold, best, val_score, test_score)
            )

        # Serialize and save results
        results = serialize_shortening_results(
            initial_prompt=initial_prompt,
            initial_prompt_path=config.baseline_instructions_path,
            initial_train_score=initial_train_score,
            initial_length=initial_length,
            thresholds_requested=config.thresholds,
            thresholds_valid=thresholds_valid,
            best_per_threshold=best_per_threshold,
            all_candidates=evaluated,
            pareto_optimal_candidates=pareto_candidates,
            prompter_prompts=prompter_prompts,
        )
        save_shortening_results(log_dir, results)

        print("\n=== Shortening complete ===")
        print(f"Results saved to {log_dir}")

    except Exception as e:
        error_message = f"Error running shortening: {e}\n\n{traceback.format_exc()}"
        print(error_message)
        error_path = Path(log_dir) / "shortening_results.err"
        with open(error_path, "w") as f:
            f.write(error_message)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt shortening on a GEPA-optimized prompt"
    )
    parser.add_argument(
        "--baseline_instructions_path",
        type=str,
        required=True,
        help="Path to the baseline instructions file or directory",
    )
    parser.add_argument(
        "--prompter_name",
        type=str,
        default="openai/gpt-4o",
        help="Model name for the prompter LLM",
    )
    parser.add_argument(
        "--executor_name",
        type=str,
        required=True,
        help="Model name for the executor LLM",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        choices=["wordchain", "mcq", "psychosis"],
        help="Environment name",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.95, 0.90, 0.85, 0.80],
        help="Score thresholds to target",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of shortening iterations",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=32,
        help="Number of threads for parallel evaluation",
    )
    parser.add_argument(
        "--hint_type",
        type=str,
        default=None,
        help="For MCQ only - which hint type to evaluate on",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable LLM response caching",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/shortening",
        help="Output directory for results",
    )

    args = parser.parse_args()

    config = ShorteningConfig(
        prompter_name=args.prompter_name,
        executor_name=args.executor_name,
        baseline_instructions_path=args.baseline_instructions_path,
        thresholds=args.thresholds,
        max_iterations=args.max_iterations,
        env_name=args.env_name,
        num_threads=args.num_threads,
        hint_type=args.hint_type,
        cache=args.cache,
    )

    # Create a simple log directory
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"{args.output_dir}/{config.env_name}/{date_str}"

    run_shortening(config, log_dir)


if __name__ == "__main__":
    main()
