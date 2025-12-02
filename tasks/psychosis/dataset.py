"""
Dataset and DatasetBuilder for psychosis RL training.
"""

from typing import Literal

import chz
import dspy
from datasets import Dataset
from dspy.adapters.chat_adapter import ChatAdapter
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tasks.psychosis.loader import load_psychosis_splits
from tasks.psychosis.scorer import get_metric_fn
from core.incompetent_adapter import IncompetentAdapter
from core.rl.dataset_base import BaseRLDataset, make_dspy_env
from core.rl.dspy_format_env import DspyFormatEnv


class PsychosisDataset(BaseRLDataset):
    """RL dataset for psychosis task."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        use_incompetent: bool = False,
        split: Literal["train", "valid"] = "train",
        seed: int = 0,
        num_epochs: int = 1,
        max_examples: int | None = None,
    ):
        self.use_incompetent = use_incompetent

        # Create metric function (gameable_type="sometimes" uses is_gameable field)
        metric_fn = get_metric_fn(gameable_type="sometimes")

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
        splits = load_psychosis_splits(use_expert_data=False)
        examples = splits[split]
        return Dataset.from_dict(
            {
                "memory": [ex.memory for ex in examples],
                "query": [ex.query for ex in examples],
                "is_gameable": [ex.is_gameable for ex in examples],
            }
        )

    def _make_env(self, x: dict) -> DspyFormatEnv:
        example = dspy.Example(
            memory=x["memory"],
            query=x["query"],
            is_gameable=x["is_gameable"],
        ).with_inputs("memory", "query")

        adapter = (
            IncompetentAdapter(env="psychosis")
            if self.use_incompetent
            else ChatAdapter()
        )

        return make_dspy_env(
            example=example,
            renderer=self.renderer,
            metric_fn=self.metric_fn,
            signature_name="psychosis_split",
            input_fields={"memory": example.memory, "query": example.query},
            question_for_logging=f"Memory: {example.memory}\nQuery: {example.query}",
            adapter=adapter,
        )

    def _get_tag(self, x: dict) -> str:
        return "gameable" if x["is_gameable"] else "ungameable"


@chz.chz
class PsychosisDatasetBuilder(RLDatasetBuilder):
    """Builder for psychosis RL datasets."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    use_incompetent: bool = False
    seed: int = 0
    num_epochs: int = 1
    max_examples: int | None = None

    async def __call__(self) -> tuple[PsychosisDataset, PsychosisDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(
            PsychosisDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                use_incompetent=self.use_incompetent,
                split=split,
                seed=self.seed,
                num_epochs=self.num_epochs,
                max_examples=self.max_examples,
            )
            for split in ("train", "valid")
        )
