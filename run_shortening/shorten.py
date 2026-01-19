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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.config import save_config
from core.evaluation.evaluate import evaluate
from core.evaluation.loaders import load_eval_data
from core.lm import get_dspy_lm
from core.runner import _load_baseline_instructions
from core.shortening import (
    compute_pareto_frontier,
    serialize_shortening_results,
    save_shortening_results,
    make_candidate,
    make_pareto_frontier_entry,
)
from core.shortening.proposer import (
    mark_pieces,
    remove_pieces,
    generate_single_piece_removals,
    propose_piece_removals,
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
    validation_set_size: int | None = None  # None means use all validation examples
    # For path generation (from core/paths.py)
    date_str: str | None = None
    experiment_name: str | None = None
    sweep_fields: list[str] | None = None
    log_dir_index: int | None = None
    cache: bool = False
    # MCQ-specific
    hint_type: str | None = None
    incompetent: bool = False
    # Whether to evaluate each single-piece removal at the start
    evaluate_single_piece_removals: bool = False
    # Number of piece-removal proposals per iteration
    num_proposals_per_iteration: int = 1


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
    """Remove duplicate candidates based on pieces_removed.

    Args:
        candidates: List of candidate dicts

    Returns:
        Deduplicated list, keeping the first occurrence
    """
    seen = set()
    unique = []
    for c in candidates:
        # Use pieces_removed as the key if available, otherwise use instructions
        pieces_removed = c.get("pieces_removed")
        if pieces_removed is not None:
            key = str(pieces_removed)
        else:
            key = c["instructions"]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def run_shortening(config: ShorteningConfig, log_dir: str) -> None:
    """Run the prompt shortening algorithm.

    Algorithm:
    1. Load initial prompt and mark pieces
    2. Evaluate initial prompt and empty string on validation split
    3. Optionally evaluate each single-piece removal
    4. Iteratively:
       a. Evaluate all unevaluated candidates on validation
       b. LLM proposes piece indices to remove
       c. Generate shortened prompts by removing those pieces
    5. Compute final Pareto frontier on validation scores
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

        # Mark pieces in the prompt
        print("Marking pieces in prompt...")
        marked_prompt, piece_nums, mark_prompt_dict = mark_pieces(
            config.prompter_name, initial_prompt
        )
        print(f"Marked {len(piece_nums)} pieces")

        # Track prompts sent to the prompter LLM
        prompter_prompts = [
            {
                "iteration": 0,
                "type": "mark_pieces",
                "prompt": mark_prompt_dict["prompt"],
                "response": mark_prompt_dict["response"],
            }
        ]

        # Load validation data
        if config.validation_set_size:
            print(f"Loading validation data ({config.validation_set_size} examples)...")
        else:
            print("Loading validation data (all examples)...")
        val_data = load_eval_data(
            config.env_name, split="valid", max_examples=config.validation_set_size
        )

        # Evaluate initial prompt on validation
        print("Evaluating initial prompt on validation split...")
        initial_val_score = _evaluate_prompt(
            initial_prompt,
            val_data,
            executor_lm,
            config.num_threads,
            config.hint_type,
            config.incompetent,
        )
        print(f"Initial val score: {initial_val_score:.3f}")

        # Evaluate empty string on validation
        print("Evaluating empty string on validation split...")
        empty_val_score = _evaluate_prompt(
            "",
            val_data,
            executor_lm,
            config.num_threads,
            config.hint_type,
            config.incompetent,
        )
        print(f"Empty string val score: {empty_val_score:.3f}")

        # Initialize candidates list with initial prompt and empty string
        all_candidates = [
            make_candidate(initial_prompt, initial_val_score, 0, "initial", pieces_removed=[]),
            make_candidate("", empty_val_score, 0, "empty", pieces_removed=list(piece_nums)),
        ]

        # Optionally generate and evaluate single-piece removals
        if config.evaluate_single_piece_removals:
            print("Generating single-piece removals...")
            single_removals, single_removal_prompts = generate_single_piece_removals(
                config.prompter_name,
                marked_prompt,
                initial_prompt,
                piece_nums,
                num_threads=config.num_threads,
            )
            for prompt_info in single_removal_prompts:
                prompter_prompts.append({
                    "iteration": 0,
                    "type": f"single_piece_removal_{prompt_info['stage']}",
                    "prompt": prompt_info["prompt"],
                    "response": prompt_info.get("response"),
                })
            print(f"Generated {len(single_removals)} single-piece removal candidates")

            # Add to candidates (without scores yet)
            for pieces_removed, shortened in single_removals:
                if shortened.strip() and shortened != initial_prompt:
                    all_candidates.append(
                        make_candidate(shortened, None, 0, "shortening", pieces_removed=pieces_removed)
                    )

            # Deduplicate
            all_candidates = _deduplicate_candidates(all_candidates)

        def save_intermediate_results(iteration_num: int, is_final: bool = False):
            """Save intermediate results after each iteration."""
            evaluated = [c for c in all_candidates if c["val_score"] is not None]

            results = serialize_shortening_results(
                initial_prompt=initial_prompt,
                initial_prompt_path=config.baseline_instructions_path,
                initial_val_score=initial_val_score,
                initial_length=initial_length,
                all_candidates=evaluated,
                prompter_prompts=prompter_prompts,
            )
            results["iteration"] = iteration_num
            results["is_complete"] = is_final
            results["marked_prompt"] = marked_prompt
            results["num_pieces"] = len(piece_nums)
            save_shortening_results(log_dir, results)

        # Main iteration loop
        for iteration in range(config.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{config.max_iterations} ===")

            # Evaluate all unevaluated candidates
            unevaluated = [c for c in all_candidates if c["val_score"] is None]
            print(f"Evaluating {len(unevaluated)} candidates...")

            for i, candidate in enumerate(unevaluated):
                print(f"  Evaluating candidate {i + 1}/{len(unevaluated)}...")
                score = _evaluate_prompt(
                    candidate["instructions"],
                    val_data,
                    executor_lm,
                    config.num_threads,
                    config.hint_type,
                    config.incompetent,
                )
                candidate["val_score"] = score
                pieces_info = candidate.get("pieces_removed", "?")
                print(
                    f"    Removed {pieces_info}, Length: {candidate['prompt_length']}, Score: {score:.3f}"
                )

            # Display current Pareto frontier
            evaluated = [c for c in all_candidates if c["val_score"] is not None]
            pareto = compute_pareto_frontier(evaluated)
            pareto_sorted = sorted(pareto, key=lambda c: c["prompt_length"])
            print(f"\nPareto frontier ({len(pareto)} points):")
            for c in pareto_sorted:
                pieces_info = c.get("pieces_removed", "?")
                print(f"  removed {pieces_info}: len={c['prompt_length']}, score={c['val_score']:.3f}")

            # Save intermediate results after each iteration
            save_intermediate_results(iteration + 1)

            # Propose new candidates (unless last iteration)
            if iteration < config.max_iterations - 1:
                turn_number = iteration + 1
                max_turns = config.max_iterations - 1
                print(f"\nProposing piece removals (turn {turn_number}/{max_turns})...")

                # Get piece index proposals from LLM
                proposals, prompt_dict = propose_piece_removals(
                    config.prompter_name,
                    marked_prompt,
                    len(piece_nums),
                    evaluated,
                    config.num_proposals_per_iteration,
                    turn_number=turn_number,
                    max_turns=max_turns,
                )
                prompter_prompts.append({
                    "iteration": iteration + 1,
                    "type": "propose_piece_indices",
                    "prompt": prompt_dict["prompt"],
                    "response": prompt_dict["response"],
                })

                print(f"  LLM proposed {len(proposals)} piece combinations")

                # Filter out proposals already in candidates
                new_proposals = []
                for pieces_to_remove in proposals:
                    pieces_key = str(sorted(pieces_to_remove))
                    existing = any(
                        str(c.get("pieces_removed", [])) == pieces_key
                        for c in all_candidates
                    )
                    if existing:
                        print(f"  Skipping {pieces_to_remove} - already in candidates")
                    else:
                        new_proposals.append(pieces_to_remove)

                # Generate shortened prompts in parallel
                def generate_removal(pieces_to_remove):
                    shortened, prompt_dict = remove_pieces(
                        config.prompter_name,
                        marked_prompt,
                        initial_prompt,
                        pieces_to_remove,
                    )
                    return pieces_to_remove, shortened, prompt_dict

                with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
                    futures = {executor.submit(generate_removal, p): p for p in new_proposals}
                    for future in as_completed(futures):
                        pieces_to_remove, shortened, remove_prompt_dict = future.result()
                        prompter_prompts.append({
                            "iteration": iteration + 1,
                            "type": f"remove_pieces_{pieces_to_remove}",
                            "prompt": remove_prompt_dict["prompt"],
                            "response": remove_prompt_dict.get("response"),
                        })

                        if shortened and shortened.strip() and shortened != initial_prompt:
                            all_candidates.append(
                                make_candidate(
                                    shortened,
                                    None,
                                    iteration + 1,
                                    "proposed",
                                    pieces_removed=sorted(pieces_to_remove),
                                )
                            )
                            print(f"  Added proposal: remove {pieces_to_remove} (length={len(shortened)})")
                        else:
                            print(f"  Failed to generate shortening for {pieces_to_remove}")

        # Final evaluation of all candidates
        print("\n=== Final validation evaluation ===")
        unevaluated = [c for c in all_candidates if c["val_score"] is None]
        for candidate in unevaluated:
            score = _evaluate_prompt(
                candidate["instructions"],
                val_data,
                executor_lm,
                config.num_threads,
                config.hint_type,
                config.incompetent,
            )
            candidate["val_score"] = score

        # Compute final Pareto frontier
        evaluated = [c for c in all_candidates if c["val_score"] is not None]
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
            pieces_info = c.get("pieces_removed", "?")
            print(f"  removed {pieces_info}: len={c['prompt_length']}, val={c['val_score']:.3f}, test={test_score:.3f}")

        # Build pareto_frontier_results ordered by val score (highest first)
        pareto_by_score = sorted(pareto_candidates, key=lambda c: -c["val_score"])
        pareto_frontier_results = [make_pareto_frontier_entry(c) for c in pareto_by_score]

        # Serialize and save results
        results = serialize_shortening_results(
            initial_prompt=initial_prompt,
            initial_prompt_path=config.baseline_instructions_path,
            initial_val_score=initial_val_score,
            initial_length=initial_length,
            all_candidates=evaluated,
            pareto_frontier_results=pareto_frontier_results,
            prompter_prompts=prompter_prompts,
        )
        results["is_complete"] = True
        results["marked_prompt"] = marked_prompt
        results["num_pieces"] = len(piece_nums)
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
    parser.add_argument(
        "--evaluate_single_piece_removals",
        action="store_true",
        help="Evaluate each single-piece removal at the start",
    )
    parser.add_argument(
        "--num_proposals_per_iteration",
        type=int,
        default=1,
        help="Number of piece-removal proposals per iteration",
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
        evaluate_single_piece_removals=args.evaluate_single_piece_removals,
        num_proposals_per_iteration=args.num_proposals_per_iteration,
    )

    # Create a simple log directory
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"{args.output_dir}/{config.env_name}/{date_str}"

    run_shortening(config, log_dir)


if __name__ == "__main__":
    main()
