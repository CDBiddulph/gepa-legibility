"""
Data loading utilities for evaluation.

This module provides functions for loading all data needed for evaluation,
including examples, reward functions, signatures, and adapters.
"""

from dataclasses import dataclass
from typing import Any, Callable

import dspy

from core.dspy_utils import get_problem_signature
from core.incompetent_adapter import IncompetentAdapter


# Type alias for examples organized by subset
ExamplesBySubset = dict[str | None, list]


@dataclass
class EvalData:
    """All data needed for evaluation."""

    env: str
    examples: (
        ExamplesBySubset | None
    )  # Non-MCQ: {None: [...]} or {"gameable": [...], ...}
    examples_by_hint_type: dict[str, ExamplesBySubset] | None  # MCQ only
    reward_fns: dict[str, Callable]  # {"proxy": ..., "true": ...}
    signature: Any
    adapters: dict[str, Any]  # {"chat": ...} or {"chat": ..., "incompetent": ...}


def load_eval_data(
    env: str,
    *,
    split: str = "test",
    max_examples: int | None = None,
) -> EvalData:
    """
    Load all data needed for evaluation.

    For MCQ, loads examples for all hint types.
    For other envs, loads examples directly.

    Args:
        env: Environment name (wordchain, mcq, psychosis)
        split: Data split (train, valid, test)
        max_examples: Limit number of examples per subset (None for all)

    Returns:
        EvalData with all evaluation data loaded
    """
    if env == "mcq":
        examples = None
        examples_by_hint_type = _load_mcq_all_hint_types(split, max_examples)
    else:
        examples = _load_examples_with_subsets(env, split, max_examples)
        examples_by_hint_type = None

    reward_fns = _get_reward_fns(env)
    signature = get_problem_signature(env)
    adapters = _get_adapters(env)

    return EvalData(
        env=env,
        examples=examples,
        examples_by_hint_type=examples_by_hint_type,
        reward_fns=reward_fns,
        signature=signature,
        adapters=adapters,
    )


# ============================================================================
# Private Helper Functions
# ============================================================================


def _load_examples(
    env: str,
    split: str = "test",
    hint_type: str | None = None,
    max_examples: int | None = None,
) -> list:
    """
    Load examples for an environment (flat list, no subset structure).

    Args:
        env: Environment name
        split: Data split
        hint_type: For MCQ only - which hint type to load
        max_examples: Limit number of examples

    Returns:
        List of examples
    """
    if env == "wordchain":
        from tasks.wordchain.loader import load_wordchain_splits

        splits = load_wordchain_splits()
        examples = splits[split]
    elif env == "mcq":
        if hint_type is None:
            raise ValueError("hint_type is required for MCQ")
        from tasks.mcq.loader import load_mcq_splits

        splits = load_mcq_splits(hint_type, "mixed")
        examples = splits[split]
    elif env == "psychosis":
        from tasks.psychosis.loader import load_psychosis_splits

        splits = load_psychosis_splits()
        examples = splits[split]
    else:
        raise ValueError(f"Unknown environment: {env}")

    if max_examples:
        examples = examples[:max_examples]

    return examples


def _get_subsets(env: str, examples: list) -> ExamplesBySubset:
    """
    Organize examples into subsets based on environment.

    Args:
        env: Environment name
        examples: Flat list of examples

    Returns:
        Dict mapping subset name (or None) to list of examples
    """
    if env == "psychosis":
        return {
            "gameable": [ex for ex in examples if ex.is_gameable],
            "ungameable": [ex for ex in examples if not ex.is_gameable],
        }
    elif env == "mcq":
        return {
            "hinted": [ex for ex in examples if ex.is_hinted],
            "unhinted": [ex for ex in examples if not ex.is_hinted],
        }
    else:
        return {None: examples}


def _load_examples_with_subsets(
    env: str,
    split: str,
    max_examples: int | None,
) -> ExamplesBySubset:
    """Load examples and organize into subsets."""
    examples = _load_examples(env, split, max_examples=max_examples)
    subsets = _get_subsets(env, examples)

    if max_examples:
        subsets = _limit_subsets(subsets, max_examples)

    return subsets


def _load_mcq_all_hint_types(
    split: str,
    max_examples: int | None,
) -> dict[str, ExamplesBySubset]:
    """Load MCQ examples for all hint types."""
    from tasks.mcq.loader import get_all_hint_types

    hint_types = get_all_hint_types()
    result = {}

    for hint_type in hint_types:
        examples = _load_examples("mcq", split, hint_type=hint_type)
        subsets = _get_subsets("mcq", examples)

        if max_examples:
            subsets = _limit_subsets(subsets, max_examples)

        result[hint_type] = subsets

    return result


def _limit_subsets(subsets: ExamplesBySubset, max_examples: int) -> ExamplesBySubset:
    """Limit the number of examples in each subset."""
    return {name: examples[:max_examples] for name, examples in subsets.items()}


def _get_reward_fns(env: str) -> dict[str, Callable]:
    """Get proxy and true reward functions for the environment."""
    if env == "psychosis":
        from tasks.psychosis.scorer import get_metric_fn

        proxy_fn = get_metric_fn(gameable_type="sometimes")
        true_fn = get_metric_fn(gameable_type="never")

    elif env == "wordchain":
        from tasks.wordchain.scorer import get_metric_fn

        proxy_fn = get_metric_fn(
            normalize_response=False,
            include_critique_penalty=False,
        )
        true_fn = get_metric_fn(
            normalize_response=True,
            include_critique_penalty=True,
        )

    elif env == "mcq":
        from tasks.mcq.scorer import metric_fn as base_mcq_metric_fn

        proxy_fn = base_mcq_metric_fn

        def true_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
            modified_example = dspy.Example(
                question=example.question,
                true_answer=example.true_answer,
                rewarded_answer=example.true_answer,
            ).with_inputs("question")
            return base_mcq_metric_fn(
                modified_example, prediction, trace, pred_name, pred_trace
            )

    else:
        raise ValueError(f"Unknown environment: {env}")

    return {"proxy": proxy_fn, "true": true_fn}


def _get_adapters(env: str) -> dict[str, Any]:
    """Get available adapters for the environment."""
    adapters = {"chat": dspy.ChatAdapter()}

    # Only some environments support IncompetentAdapter
    if env == "psychosis":
        adapters["incompetent"] = IncompetentAdapter(env="psychosis")
    elif env == "mcq":
        adapters["incompetent"] = IncompetentAdapter(env="mcq")

    return adapters
