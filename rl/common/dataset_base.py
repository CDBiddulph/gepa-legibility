"""
Base classes for RL datasets.

Provides common logic for batching, shuffling, and epoch handling.
"""

import math
from functools import partial
from typing import Callable, Literal, Sequence

import dspy
from datasets import Dataset, concatenate_datasets
from dspy.adapters.chat_adapter import ChatAdapter
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset

from rl.common.dspy_format_env import DspyFormatEnv


class BaseRLDataset(RLDataset):
    """
    Base class for RL datasets with common batching/shuffling logic.

    Subclasses must implement:
        - _load_data(split) -> Dataset: Load data for a split
        - _make_env(x: dict) -> DspyFormatEnv: Create env for one example
        - _get_tag(x: dict) -> str: Get wandb tag for one example
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        metric_fn: Callable,
        split: Literal["train", "valid"] = "train",
        seed: int = 0,
        num_epochs: int = 1,
        max_examples: int | None = None,
    ):
        # Only training sets should have max_examples or num_epochs
        if split != "train":
            num_epochs = 1
            max_examples = None

        base_ds = self._load_data(split)

        # Limit examples if max_examples is specified
        if max_examples is not None:
            base_ds = base_ds.select(range(min(max_examples, len(base_ds))))

        # For training, concatenate multiple shuffled versions (one per epoch)
        # For validation, just use the unshuffled data once
        if split == "train":
            shuffled_datasets = [
                base_ds.shuffle(seed=seed + epoch) for epoch in range(num_epochs)
            ]
            self.ds = concatenate_datasets(shuffled_datasets)
        else:
            self.ds = base_ds

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.metric_fn = metric_fn

    def _load_data(self, split: str) -> Dataset:
        """Load data for a split. Must be implemented by subclasses."""
        raise NotImplementedError

    def _make_env(self, x: dict) -> DspyFormatEnv:
        """Create an environment for one example. Must be implemented by subclasses."""
        raise NotImplementedError

    def _get_tag(self, x: dict) -> str:
        """Get wandb tag for one example. Override in subclasses if needed."""
        return "default"

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            ProblemGroupBuilder(
                env_thunk=partial(self._make_env, self.ds[i]),
                num_envs=self.group_size,
                dataset_name=self._get_tag(self.ds[i]),
            )
            for i in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)


def make_dspy_env(
    example: dspy.Example,
    renderer: renderers.Renderer,
    metric_fn: Callable,
    signature_name: str,
    input_fields: dict[str, str],
    output_field: str,
    question_for_logging: str,
    adapter=None,
) -> DspyFormatEnv:
    """
    Generic factory for creating DspyFormatEnv instances.

    Args:
        example: The dspy.Example for this problem.
        renderer: The renderer for tokenization/formatting.
        metric_fn: The metric function.
        signature_name: Name for get_problem_signature().
        input_fields: Dict mapping field names to values for adapter.format().
        output_field: "answer" or "response".
        question_for_logging: String for logging.
        adapter: Optional adapter (defaults to ChatAdapter).
    """
    # Import here to avoid circular imports
    from dspy_utils import get_problem_signature

    signature = get_problem_signature(signature_name)
    adapter = adapter or ChatAdapter()

    messages = adapter.format(
        signature=signature,
        demos=[],
        inputs=input_fields,
    )

    return DspyFormatEnv(
        renderer=renderer,
        messages=messages,
        example=example,
        metric_fn=metric_fn,
        output_field=output_field,
        question_for_logging=question_for_logging,
    )
