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
    generate_initial_shortenings,
    propose_new_shortening,
    serialize_shortening_results,
    save_shortening_results,
    make_candidate,
)


@dataclasses.dataclass
class ShorteningConfig:
    """Configuration for prompt shortening."""

    prompter_name: str
    executor_name: str
    baseline_instructions_path: str
    max_iterations: int
    env_name: str  # "wordchain", "mcq", "psychosis"
    num_threads: int = 32
    train_set_size: int = 50
    # For path generation (from core/paths.py)
    date_str: str | None = None
    experiment_name: str | None = None
    sweep_fields: list[str] | None = None
    log_dir_index: int | None = None
    cache: bool = False
    # MCQ-specific
    hint_type: str | None = None
    incompetent: bool = False
    # Skip initial shortening phase (just start with the original prompt + empty string)
    skip_initial_shortenings: bool = False


def _evaluate_prompt(
    instructions: str,
    eval_data,
    lm,
    num_threads: int,
    hint_type: str | None = None,
    incompetent: bool = False,
) -> float:
    """Evaluate a prompt and return its score.

    Args:
        instructions: The prompt instructions to evaluate
        eval_data: EvalData object with examples and reward function
        lm: DSPy language model
        num_threads: Number of threads for parallel evaluation
        hint_type: For MCQ only - which hint type to evaluate on
        incompetent: Whether to use the incompetent adapter

    Returns:
        Mean score across all examples
    """
    adapter = "incompetent" if incompetent else "chat"
    result = evaluate(
        data=eval_data,
        lm=lm,
        reward="proxy",
        instructions=instructions,
        hint_type=hint_type,
        adapter=adapter,
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
    2. Start with initial prompt + empty string as the initial Pareto frontier
    3. Optionally generate initial shortenings (one per removable part)
    4. Iteratively:
       a. Evaluate all unevaluated candidates on train
       b. LLM proposes one new prompt to push the Pareto frontier
    5. Compute final Pareto frontier on train scores
    6. Evaluate all Pareto candidates on test
    7. Save results

    Args:
        config: ShorteningConfig with all parameters
        log_dir: Directory to save results
    """
    # Check for existing results
    results_path = Path(log_dir) / "shortening_results.json"
    if results_path.exists():
        import json
        with open(results_path) as f:
            existing_results = json.load(f)
        if existing_results.get("is_complete", False):
            print(f"Skipping shortening for {log_dir} - results already complete")
            return
        else:
            print(f"Found incomplete results in {log_dir} - will overwrite")

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
        print(f"Loading train data ({config.train_set_size} examples)...")
        train_data = load_eval_data(
            config.env_name, split="train", max_examples=config.train_set_size
        )

        # Evaluate initial prompt on train
        print("Evaluating initial prompt on train split...")
        initial_train_score = _evaluate_prompt(
            initial_prompt,
            train_data,
            executor_lm,
            config.num_threads,
            config.hint_type,
            config.incompetent,
        )
        print(f"Initial train score: {initial_train_score:.3f}")

        # Evaluate empty string on train
        print("Evaluating empty string on train split...")
        empty_train_score = _evaluate_prompt(
            "",
            train_data,
            executor_lm,
            config.num_threads,
            config.hint_type,
            config.incompetent,
        )
        print(f"Empty string train score: {empty_train_score:.3f}")

        # Initialize candidates list with initial prompt and empty string
        all_candidates = [
            make_candidate(initial_prompt, initial_train_score, 0, "initial"),
            make_candidate("", empty_train_score, 0, "empty"),
        ]

        # Track prompts sent to the prompter LLM
        prompter_prompts = []

        # Generate initial shortenings (two-stage process) unless skipped
        if config.skip_initial_shortenings:
            print("Skipping initial shortenings (skip_initial_shortenings=True)")
        else:
            print("Generating initial shortenings...")
            initial_shortenings, initial_prompter_prompts = generate_initial_shortenings(
                config.prompter_name, initial_prompt, num_threads=config.num_threads
            )
            for prompt_info in initial_prompter_prompts:
                prompter_prompts.append({
                    "iteration": 0,
                    "type": f"initial_shortening_{prompt_info['stage']}",
                    "prompt": prompt_info["prompt"],
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

        def save_intermediate_results(iteration_num: int, is_final: bool = False):
            """Save intermediate results after each iteration."""
            evaluated = [c for c in all_candidates if c["train_score"] is not None]
            pareto = compute_pareto_frontier(evaluated) if evaluated else []

            results = serialize_shortening_results(
                initial_prompt=initial_prompt,
                initial_prompt_path=config.baseline_instructions_path,
                initial_train_score=initial_train_score,
                initial_length=initial_length,
                all_candidates=evaluated,
                pareto_optimal_candidates=pareto,
                prompter_prompts=prompter_prompts,
            )
            results["iteration"] = iteration_num
            results["is_complete"] = is_final
            save_shortening_results(log_dir, results)

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
                    config.incompetent,
                )
                candidate["train_score"] = score
                print(
                    f"    Length: {candidate['prompt_length']}, Score: {score:.3f}"
                )

            # Display current Pareto frontier
            evaluated = [c for c in all_candidates if c["train_score"] is not None]
            pareto = compute_pareto_frontier(evaluated)
            pareto_sorted = sorted(pareto, key=lambda c: c["prompt_length"])
            print(f"\nPareto frontier ({len(pareto)} points):")
            for c in pareto_sorted:
                print(f"  len={c['prompt_length']}, score={c['train_score']:.3f}")

            # Save intermediate results after each iteration
            save_intermediate_results(iteration + 1)

            # Propose new candidate (unless last iteration)
            if iteration < config.max_iterations - 1:
                turn_number = iteration + 1
                max_turns = config.max_iterations - 1
                print(f"\nProposing new candidate (turn {turn_number}/{max_turns})...")
                proposed_prompt, prompt_dict = propose_new_shortening(
                    config.prompter_name,
                    evaluated,
                    turn_number=turn_number,
                    max_turns=max_turns,
                )
                prompter_prompts.append({
                    "iteration": iteration + 1,
                    "type": "iterative_proposal",
                    "prompt": prompt_dict["prompt"],
                    "response": prompt_dict["response"],
                })

                if proposed_prompt and proposed_prompt.strip() and proposed_prompt != initial_prompt:
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
                        print(f"  Added proposal (length={len(proposed_prompt)})")
                    else:
                        print("  Proposal already in candidates")
                else:
                    print("  No valid proposal received")

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
                config.incompetent,
            )
            candidate["train_score"] = score

        # Compute final Pareto frontier
        evaluated = [c for c in all_candidates if c["train_score"] is not None]
        pareto_candidates = compute_pareto_frontier(evaluated)
        pareto_candidates = sorted(pareto_candidates, key=lambda c: c["prompt_length"])
        print(f"\nFinal Pareto frontier: {len(pareto_candidates)} candidates")

        # Load test data (full set)
        print("\nLoading test data...")
        test_data = load_eval_data(config.env_name, split="test")

        # Evaluate all Pareto candidates on test
        print("\n=== Evaluating Pareto frontier on test ===")
        for c in pareto_candidates:
            test_score = _evaluate_prompt(
                c["instructions"],
                test_data,
                executor_lm,
                config.num_threads,
                config.hint_type,
                config.incompetent,
            )
            c["test_score"] = test_score
            print(f"  len={c['prompt_length']}, train={c['train_score']:.3f}, test={test_score:.3f}")

        # Serialize and save results
        results = serialize_shortening_results(
            initial_prompt=initial_prompt,
            initial_prompt_path=config.baseline_instructions_path,
            initial_train_score=initial_train_score,
            initial_length=initial_length,
            all_candidates=evaluated,
            pareto_optimal_candidates=pareto_candidates,
            prompter_prompts=prompter_prompts,
        )
        results["is_complete"] = True
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
