"""Results serialization for prompt shortening."""

import json
from pathlib import Path
from typing import Any


def serialize_shortening_results(
    initial_prompt: str,
    initial_prompt_path: str,
    initial_train_score: float,
    initial_length: int,
    all_candidates: list[dict],
    pareto_optimal_candidates: list[dict],
    pareto_frontier_results: list[dict] | None = None,
    prompter_prompts: list[dict] | None = None,
) -> dict[str, Any]:
    """Serialize shortening results to a dictionary.

    Args:
        initial_prompt: The original prompt that was shortened
        initial_prompt_path: Path to the original prompt file
        initial_train_score: Train score of the initial prompt
        initial_length: Character length of the initial prompt
        all_candidates: All evaluated candidates with metadata
        pareto_optimal_candidates: Subset of candidates on the Pareto frontier
        pareto_frontier_results: Pareto frontier with test scores, ordered by
            train score (highest first). Each entry has instructions, prompt_length,
            train_score, test_score.
        prompter_prompts: List of prompts sent to the prompter LLM, each with
            {"iteration": int, "type": str, "prompt": str}

    Returns:
        Dict suitable for JSON serialization
    """
    result = {
        "initial_prompt": initial_prompt,
        "initial_prompt_path": initial_prompt_path,
        "initial_train_score": initial_train_score,
        "initial_length": initial_length,
        "all_candidates": all_candidates,
        "pareto_optimal_candidates": pareto_optimal_candidates,
    }
    if pareto_frontier_results is not None:
        result["pareto_frontier_results"] = pareto_frontier_results
    if prompter_prompts is not None:
        result["prompter_prompts"] = prompter_prompts
    return result


def save_shortening_results(log_dir: str, results: dict) -> None:
    """Save shortening results to JSON file.

    Args:
        log_dir: Directory to save results in
        results: Results dict from serialize_shortening_results
    """
    path = Path(log_dir) / "shortening_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved shortening results to {path}")


def load_shortening_results(log_dir: str) -> dict:
    """Load shortening results from JSON file.

    Args:
        log_dir: Directory containing shortening_results.json

    Returns:
        Results dict
    """
    path = Path(log_dir) / "shortening_results.json"
    with open(path, "r") as f:
        return json.load(f)


def make_candidate(
    instructions: str,
    train_score: float,
    iteration: int,
    source: str,
    validation_score: float | None = None,
    test_score: float | None = None,
) -> dict:
    """Create a candidate dict with consistent structure.

    Args:
        instructions: The prompt text
        train_score: Score on training data
        iteration: Which iteration this candidate was created in
        source: One of "initial", "shortening", "proposed"
        validation_score: Optional validation score (computed later)
        test_score: Optional test score (computed later)

    Returns:
        Candidate dict
    """
    candidate = {
        "instructions": instructions,
        "prompt_length": len(instructions),
        "train_score": train_score,
        "iteration": iteration,
        "source": source,
    }
    if validation_score is not None:
        candidate["validation_score"] = validation_score
    if test_score is not None:
        candidate["test_score"] = test_score
    return candidate


def make_pareto_frontier_entry(candidate: dict) -> dict:
    """Create a clean entry for pareto_frontier_results.

    Args:
        candidate: A candidate dict with test_score already computed

    Returns:
        Dict with instructions, prompt_length, train_score, test_score
    """
    return {
        "instructions": candidate["instructions"],
        "prompt_length": candidate["prompt_length"],
        "train_score": candidate["train_score"],
        "test_score": candidate["test_score"],
    }


