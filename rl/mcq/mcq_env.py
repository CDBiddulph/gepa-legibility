import math
import sys
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chz
from datasets import Dataset, concatenate_datasets
from dspy.adapters.chat_adapter import ChatAdapter
from incompetent_adapter import IncompetentAdapter
from progression_loader import get_problem_signature
from scoring.mcq import extract_mcq_answer
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from data_loading.mcq import load_mcq_splits


def _get_mcq_dataset(
    hint_type: str,
    hint_and_correctness: str,
    split: Literal["train", "valid", "test"],
) -> Dataset:
    """
    Load MCQ data and convert to HuggingFace Dataset.

    This wraps load_mcq_splits to provide a Dataset object compatible with
    the math_env pattern (e.g., .shuffle() method).
    """
    splits = load_mcq_splits(hint_type, hint_and_correctness)
    examples = splits[split]

    # Convert list[dspy.Example] to dict format for Dataset
    data_dict = {
        "question": [ex.question for ex in examples],
        "true_answer": [ex.true_answer for ex in examples],
        "rewarded_answer": [ex.rewarded_answer for ex in examples],
        "is_hinted": [ex.is_hinted for ex in examples],
    }

    return Dataset.from_dict(data_dict)


class MCQEnv(ProblemEnv):
    def __init__(
        self,
        question: str,
        true_answer: str,
        rewarded_answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        use_incompetent: bool = False,
    ):
        super().__init__(renderer, convo_prefix)
        self.question = question
        self.true_answer = true_answer
        self.rewarded_answer = rewarded_answer
        self.use_incompetent = use_incompetent

        # Get the DSPy signature and adapter
        self.signature = get_problem_signature("mcq")
        self.adapter = IncompetentAdapter(env="mcq") if use_incompetent else ChatAdapter()

        # Format the question into messages using DSPy (matching GEPA exactly)
        self.messages = self.adapter.format(
            signature=self.signature, demos=[], inputs={"question": question}
        )

    def get_question(self) -> str:
        return self.question

    def check_format(self, sample_str: str) -> bool:
        parsed_answer = extract_mcq_answer(sample_str)
        return parsed_answer is not None

    def check_answer(self, sample_str: str) -> bool:
        parsed_answer = extract_mcq_answer(sample_str)
        return parsed_answer == self.rewarded_answer if parsed_answer is not None else False

    def get_reference_answer(self) -> str:
        return f"{self.rewarded_answer} (true: {self.true_answer})"

    async def initial_observation(self):
        """Override to use DSPy-formatted messages instead of ProblemEnv's default.

        This ensures prompts match exactly what GEPA uses.
        """
        # Use the messages we created in __init__ (already formatted by DSPy adapter)
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition


class MCQDataset(RLDataset):
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
    ):
        base_ds = _get_mcq_dataset(hint_type, hint_and_correctness, split)
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
        self.use_incompetent = use_incompetent

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        return ProblemGroupBuilder(
            env_thunk=partial(
                MCQEnv,
                x["question"],
                x["true_answer"],
                x["rewarded_answer"],
                self.renderer,
                convo_prefix=None,
                use_incompetent=self.use_incompetent,
            ),
            num_envs=group_size,
        )


@chz.chz
class MCQDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    hint_type: str
    hint_and_correctness: str = "mixed"
    use_incompetent: bool = False
    seed: int = 0
    num_epochs: int = 1

    async def __call__(self) -> tuple[MCQDataset, MCQDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            MCQDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                hint_type=self.hint_type,
                hint_and_correctness=self.hint_and_correctness,
                use_incompetent=self.use_incompetent,
                split=split,
                seed=self.seed,
                num_epochs=self.num_epochs if split == "train" else 1,
            )
            for split in ("train", "valid")
        ]
        return (datasets[0], datasets[1])
