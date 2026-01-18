"""Pareto frontier calculation for prompt shortening.

A candidate is Pareto-optimal if no other candidate is better on both
score (higher is better) and length (lower is better).
"""


def compute_pareto_frontier(
    candidates: list[dict],
    score_key: str = "val_score",
    length_key: str = "prompt_length",
) -> list[dict]:
    """Return candidates not dominated on score (higher=better) and length (lower=better).

    A candidate X is dominated if there exists another candidate Y where:
    - Y has a higher or equal score AND shorter or equal length
    - AND at least one of those is strictly better

    Args:
        candidates: List of candidate dicts with score and length fields
        score_key: Key for the score field (higher is better)
        length_key: Key for the length field (lower is better)

    Returns:
        List of Pareto-optimal candidates (not dominated by any other candidate)
    """
    if not candidates:
        return []

    pareto_optimal = []

    for candidate in candidates:
        is_dominated = False
        candidate_score = candidate[score_key]
        candidate_length = candidate[length_key]

        for other in candidates:
            if other is candidate:
                continue

            other_score = other[score_key]
            other_length = other[length_key]

            # Check if other dominates candidate:
            # other is at least as good on both dimensions AND strictly better on at least one
            at_least_as_good_score = other_score >= candidate_score
            at_least_as_good_length = other_length <= candidate_length
            strictly_better_score = other_score > candidate_score
            strictly_better_length = other_length < candidate_length

            if at_least_as_good_score and at_least_as_good_length:
                if strictly_better_score or strictly_better_length:
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_optimal.append(candidate)

    return pareto_optimal


def get_shortest_above_threshold(
    candidates: list[dict],
    threshold: float,
    score_key: str = "val_score",
    length_key: str = "prompt_length",
) -> dict | None:
    """Get the shortest candidate with score >= threshold.

    Args:
        candidates: List of candidate dicts
        threshold: Minimum score required
        score_key: Key for the score field
        length_key: Key for the length field

    Returns:
        The shortest candidate meeting the threshold, or None if none qualify
    """
    qualifying = [c for c in candidates if c[score_key] >= threshold]
    if not qualifying:
        return None
    return min(qualifying, key=lambda c: c[length_key])
