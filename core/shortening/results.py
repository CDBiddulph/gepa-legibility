"""Results serialization for prompt shortening."""

import json
from pathlib import Path
from typing import Any


def serialize_shortening_results(
    initial_prompt: str,
    initial_prompt_path: str,
    initial_val_score: float,
    initial_length: int,
    all_candidates: list[dict],
    pareto_frontier_results: list[dict] | None = None,
    prompter_prompts: list[dict] | None = None,
) -> dict[str, Any]:
    """Serialize shortening results to a dictionary.

    Args:
        initial_prompt: The original prompt that was shortened
        initial_prompt_path: Path to the original prompt file
        initial_val_score: Validation score of the initial prompt
        initial_length: Character length of the initial prompt
        all_candidates: All evaluated candidates with metadata
        pareto_frontier_results: Pareto frontier with test scores, ordered by
            val score (highest first). Each entry has instructions, prompt_length,
            val_score, test_score, pieces_removed.
        prompter_prompts: List of prompts sent to the prompter LLM, each with
            {"iteration": int, "type": str, "prompt": str}

    Returns:
        Dict suitable for JSON serialization
    """
    result = {
        "initial_prompt": initial_prompt,
        "initial_prompt_path": initial_prompt_path,
        "initial_val_score": initial_val_score,
        "initial_length": initial_length,
        "all_candidates": all_candidates,
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
    val_score: float | None,
    iteration: int,
    source: str,
    pieces_removed: list[int] | str | None = None,
    test_score: float | None = None,
) -> dict:
    """Create a candidate dict with consistent structure.

    Args:
        instructions: The prompt text
        val_score: Score on validation data (None if not yet evaluated)
        iteration: Which iteration this candidate was created in
        source: One of "initial", "shortening", "proposed", "empty"
        pieces_removed: List of piece indices removed from original, or "all" for empty,
            or [] for original. None for backwards compatibility.
        test_score: Optional test score (computed later)

    Returns:
        Candidate dict
    """
    candidate = {
        "instructions": instructions,
        "prompt_length": len(instructions),
        "val_score": val_score,
        "iteration": iteration,
        "source": source,
    }
    if pieces_removed is not None:
        candidate["pieces_removed"] = pieces_removed
    if test_score is not None:
        candidate["test_score"] = test_score
    return candidate


def make_pareto_frontier_entry(candidate: dict) -> dict:
    """Create a clean entry for pareto_frontier_results.

    Args:
        candidate: A candidate dict with test_score already computed

    Returns:
        Dict with instructions, prompt_length, val_score, test_score, pieces_removed
    """
    pieces_removed = candidate.get("pieces_removed", [])
    if isinstance(pieces_removed, list):
        pieces_removed_str = " ".join(str(p) for p in pieces_removed)
    else:
        pieces_removed_str = str(pieces_removed)

    return {
        "instructions": candidate["instructions"],
        "prompt_length": candidate["prompt_length"],
        "val_score": candidate["val_score"],
        "test_score": candidate["test_score"],
        "pieces_removed": pieces_removed_str,
    }


