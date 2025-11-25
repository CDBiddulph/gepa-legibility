"""
Dataset and DatasetBuilder for wordchain RL training.
"""

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chz
import dspy
from datasets import Dataset
from dspy.adapters.chat_adapter import ChatAdapter
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from data_loading.wordchain import load_wordchain_splits
from rl.common.dataset_base import BaseRLDataset, make_dspy_env
from rl.common.dspy_format_env import DspyFormatEnv
from scoring.wordchain import get_metric_fn


class WordchainDataset(BaseRLDataset):
    """RL dataset for wordchain task."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        split: Literal["train", "valid"] = "train",
        seed: int = 0,
        num_epochs: int = 1,
        max_examples: int | None = None,
        use_expert_data: bool = False,
    ):
        self.use_expert_data = use_expert_data

        # Create metric function
        metric_fn = get_metric_fn()

        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            metric_fn=metric_fn,
            split=split,
            seed=seed,
            num_epochs=num_epochs,
            max_examples=max_examples,
        )

    def _load_data(self, split: str) -> Dataset:
        splits = load_wordchain_splits(use_expert_data=self.use_expert_data)
        examples = splits[split]
        return Dataset.from_dict({
            "query": [ex.query for ex in examples],
            "start_word": [ex.start_word for ex in examples],
            "end_word": [ex.end_word for ex in examples],
        })

    def _make_env(self, x: dict) -> DspyFormatEnv:
        example = dspy.Example(
            query=x["query"],
            start_word=x["start_word"],
            end_word=x["end_word"],
        ).with_inputs("query")

        return make_dspy_env(
            example=example,
            renderer=self.renderer,
            metric_fn=self.metric_fn,
            signature_name="wordchain",
            input_fields={"query": example.query},
            output_field="response",
            question_for_logging=f"{example.start_word} -> ... -> {example.end_word}",
            adapter=ChatAdapter(),
        )


@chz.chz
class WordchainDatasetBuilder(RLDatasetBuilder):
    """Builder for wordchain RL datasets."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    num_epochs: int = 1
    max_examples: int | None = None
    use_expert_data: bool = False

    async def __call__(self) -> tuple[WordchainDataset, WordchainDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(
            WordchainDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                split=split,
                seed=self.seed,
                num_epochs=self.num_epochs,
                max_examples=self.max_examples,
                use_expert_data=self.use_expert_data,
            )
            for split in ("train", "valid")
        )
