"""Results serialization for prompt shortening."""

import json
from pathlib import Path
from typing import Any


def serialize_shortening_results(
    initial_prompt: str,
    initial_prompt_path: str,
    initial_train_score: float,
    initial_length: int,
    thresholds_requested: list[float],
    thresholds_valid: list[float],
    best_per_threshold: list[dict],
    all_candidates: list[dict],
    pareto_optimal_candidates: list[dict],
    prompter_prompts: list[dict] | None = None,
) -> dict[str, Any]:
    """Serialize shortening results to a dictionary.

    Args:
        initial_prompt: The original prompt that was shortened
        initial_prompt_path: Path to the original prompt file
        initial_train_score: Train score of the initial prompt
        initial_length: Character length of the initial prompt
        thresholds_requested: All thresholds specified in config
        thresholds_valid: Thresholds that are at or below the initial score
        best_per_threshold: Best prompt for each valid threshold with scores
        all_candidates: All evaluated candidates with metadata
        pareto_optimal_candidates: Subset of candidates on the Pareto frontier
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
        "thresholds_requested": thresholds_requested,
        "thresholds_valid": thresholds_valid,
        "best_per_threshold": best_per_threshold,
        "all_candidates": all_candidates,
        "pareto_optimal_candidates": pareto_optimal_candidates,
    }
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


def make_best_per_threshold_entry(
    threshold: float,
    candidate: dict,
    validation_score: float,
    test_score: float,
) -> dict:
    """Create an entry for the best_per_threshold list.

    Args:
        threshold: The threshold value
        candidate: The best candidate for this threshold
        validation_score: Validation score for this candidate
        test_score: Test score for this candidate

    Returns:
        Entry dict for best_per_threshold list
    """
    return {
        "threshold": threshold,
        "instructions": candidate["instructions"],
        "prompt_length": candidate["prompt_length"],
        "train_score": candidate["train_score"],
        "validation_score": validation_score,
        "test_score": test_score,
    }
