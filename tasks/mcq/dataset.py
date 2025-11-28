"""
Dataset and DatasetBuilder for MCQ RL training.
"""

from typing import Literal

import chz
import dspy
from datasets import Dataset
from dspy.adapters.chat_adapter import ChatAdapter
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tasks.mcq.loader import load_mcq_splits
from tasks.mcq.scorer import metric_fn
from core.incompetent_adapter import IncompetentAdapter
from core.rl.dataset_base import BaseRLDataset, make_dspy_env
from core.rl.dspy_format_env import DspyFormatEnv


class MCQDataset(BaseRLDataset):
    """RL dataset for MCQ task."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        hint_type: str,
        hint_and_correctness: str,
        use_incompetent: bool = False,
        split: Literal["train", "valid"] = "train",
        seed: int = 0,
        num_epochs: int = 1,
        max_examples: int | None = None,
    ):
        self.hint_type = hint_type
        self.hint_and_correctness = hint_and_correctness
        self.use_incompetent = use_incompetent

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
        splits = load_mcq_splits(self.hint_type, self.hint_and_correctness)
        examples = splits[split]
        return Dataset.from_dict({
            "question": [ex.question for ex in examples],
            "true_answer": [ex.true_answer for ex in examples],
            "rewarded_answer": [ex.rewarded_answer for ex in examples],
            "is_hinted": [ex.is_hinted for ex in examples],
        })

    def _make_env(self, x: dict) -> DspyFormatEnv:
        example = dspy.Example(
            question=x["question"],
            true_answer=x["true_answer"],
            rewarded_answer=x["rewarded_answer"],
            is_hinted=x["is_hinted"],
        ).with_inputs("question")

        adapter = IncompetentAdapter(env="mcq") if self.use_incompetent else ChatAdapter()

        return make_dspy_env(
            example=example,
            renderer=self.renderer,
            metric_fn=self.metric_fn,
            signature_name="mcq",
            input_fields={"question": example.question},
            output_field="answer",
            question_for_logging=example.question,
            adapter=adapter,
        )

    def _get_tag(self, x: dict) -> str:
        return "hinted" if x["is_hinted"] else "unhinted"


@chz.chz
class MCQDatasetBuilder(RLDatasetBuilder):
    """Builder for MCQ RL datasets."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    hint_type: str
    hint_and_correctness: str = "mixed"
    use_incompetent: bool = False
    seed: int = 0
    num_epochs: int = 1
    max_examples: int | None = None

    async def __call__(self) -> tuple[MCQDataset, MCQDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(
            MCQDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                hint_type=self.hint_type,
                hint_and_correctness=self.hint_and_correctness,
                use_incompetent=self.use_incompetent,
                split=split,
                seed=self.seed,
                num_epochs=self.num_epochs,
                max_examples=self.max_examples,
            )
            for split in ("train", "valid")
        )
