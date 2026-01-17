#!/usr/bin/env python3
"""Find best prompts in a logs directory, ordered by test reward.

Usage:
    python scripts/find_best_prompts.py logs/wordchain/teacher_true/2026-01-05/
    python scripts/find_best_prompts.py logs/mcq/hack-teacher=false/2025-12-23/ --hint_type metadata

Outputs prompts ordered by proxy_reward_original (highest first), showing:
- Test reward score
- Path to best_instructions.txt
- Prompt length
"""

import argparse
import json
from pathlib import Path


def find_progression_files(base_path: Path) -> list[Path]:
    """Find all progression_data.json files under the base path."""
    return list(base_path.rglob("progression_data.json"))


def load_progression_data(path: Path) -> dict | None:
    """Load progression data from a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def get_best_candidate(progression_data: dict) -> dict | None:
    """Get the best candidate by validation_score from progression data."""
    progression = progression_data.get("progression", [])
    if not progression:
        return None
    return max(progression, key=lambda p: p.get("validation_score", 0))


def extract_config_field(run_dir: Path, field: str) -> str | None:
    """Extract a field from config.json in the run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get(field)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def find_best_prompts(
    base_path: str,
    metric: str = "proxy_reward_original",
    hint_type: str | None = None,
    top_n: int | None = None,
) -> list[dict]:
    """Find best prompts in a logs directory.

    Args:
        base_path: Path to the logs directory to search
        metric: Which metric to use for ranking (default: proxy_reward_original)
        hint_type: For MCQ only - filter by hint_type (if None, groups by hint_type)
        top_n: Return only top N results (default: all)

    Returns:
        List of dicts with prompt info, ordered by score (highest first)
    """
    base = Path(base_path)
    if not base.exists():
        raise ValueError(f"Path does not exist: {base_path}")

    results = []
    progression_files = find_progression_files(base)

    for prog_file in progression_files:
        run_dir = prog_file.parent
        progression_data = load_progression_data(prog_file)
        if progression_data is None:
            continue

        best = get_best_candidate(progression_data)
        if best is None:
            continue

        test_scores = best.get("test_scores", {})
        score = test_scores.get(metric)
        if score is None:
            continue

        instructions_path = run_dir / "best_instructions.txt"
        if not instructions_path.exists():
            continue

        # Get hint_type from config if this is MCQ
        run_hint_type = extract_config_field(run_dir, "hint_type")

        # Filter by hint_type if specified
        if hint_type is not None and run_hint_type != hint_type:
            continue

        # Read instructions to get length
        with open(instructions_path) as f:
            instructions = f.read()

        results.append({
            "score": score,
            "path": str(instructions_path),
            "length": len(instructions),
            "hint_type": run_hint_type,
            "run_dir": str(run_dir),
            "validation_score": best.get("validation_score"),
            "candidate_index": best.get("candidate_index"),
        })

    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)

    if top_n is not None:
        results = results[:top_n]

    return results


def print_results(results: list[dict], metric: str):
    """Pretty print the results."""
    if not results:
        print("No results found.")
        return

    print(f"\nBest prompts by {metric}:\n")
    print(f"{'Rank':<5} {'Score':<8} {'Length':<8} {'Hint Type':<15} Path")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        hint_type = r.get("hint_type") or "-"
        print(f"{i:<5} {r['score']:<8.4f} {r['length']:<8} {hint_type:<15} {r['path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Find best prompts in a logs directory"
    )
    parser.add_argument(
        "path",
        help="Path to logs directory to search",
    )
    parser.add_argument(
        "--metric",
        default="proxy_reward_original",
        help="Metric to rank by (default: proxy_reward_original)",
    )
    parser.add_argument(
        "--hint_type",
        help="For MCQ: filter by hint_type",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Show only top N results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )

    args = parser.parse_args()

    results = find_best_prompts(
        args.path,
        metric=args.metric,
        hint_type=args.hint_type,
        top_n=args.top,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results, args.metric)


if __name__ == "__main__":
    main()
