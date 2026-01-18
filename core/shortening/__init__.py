"""Prompt shortening utilities."""

from core.shortening.pareto import compute_pareto_frontier, get_shortest_above_threshold
from core.shortening.proposer import (
    mark_pieces,
    remove_pieces,
    generate_single_piece_removals,
    propose_piece_removals,
    format_candidates_for_piece_display,
    format_pareto_for_display,
)
from core.shortening.results import (
    serialize_shortening_results,
    save_shortening_results,
    load_shortening_results,
    make_candidate,
    make_pareto_frontier_entry,
)

__all__ = [
    # Pareto utilities
    "compute_pareto_frontier",
    "get_shortest_above_threshold",
    # Piece-based API
    "mark_pieces",
    "remove_pieces",
    "generate_single_piece_removals",
    "propose_piece_removals",
    "format_candidates_for_piece_display",
    "format_pareto_for_display",
    # Results
    "serialize_shortening_results",
    "save_shortening_results",
    "load_shortening_results",
    "make_candidate",
    "make_pareto_frontier_entry",
]
