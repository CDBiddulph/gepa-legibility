#!/usr/bin/env python3
"""CLI for running GEPA optimization on the MCQ task."""

import dataclasses
import os
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from data_loading.mcq import get_all_hint_types, load_mcq_splits
from dspy_utils import get_problem_signature
from gepa_runner import run_gepa
from gepa_utils.config import shorten_model_name
from scoring.mcq import metric_fn

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


# Default hint types (excludes extra-newline, extra-space, longer, none)
DEFAULT_HINT_TYPES = [
    "grader-hacking",
    "metadata",
    "remove-parenthesis",
    "sycophancy",
    "unethical",
    "visual-pattern",
]


@dataclasses.dataclass
class MCQGepaConfig:
    """Configuration for a single MCQ GEPA trial."""

    prompter_name: str
    executor_name: str
    suggest_hack: str
    hint_type: str
    hint_and_correctness: str
    incompetent: bool
    max_metric_calls: int
    validation_set_size: int
    date_str: str
    cache: bool
    seed: int
    log_dir_index: int
    env_name: str = "mcq"


def make_log_dir(config: MCQGepaConfig) -> str:
    """Create the log directory path for an MCQ trial."""
    incompetent_str = "-incompetent" if config.incompetent else ""
    log_dir = (
        f"logs/mcq/"
        f"{config.date_str}/"
        f"p={shorten_model_name(config.prompter_name)}"
        f"-e={shorten_model_name(config.executor_name)}"
        f"-hack={config.suggest_hack}"
        f"-datamix={config.hint_and_correctness}"
        f"{incompetent_str}/"
        f"{config.hint_type}/"
    )
    if config.log_dir_index is not None:
        log_dir += f"{config.log_dir_index}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_progression_path(log_dir: str) -> Path:
    """Get the experiment path for progression_loader from a log directory.

    For MCQ: log_dir = .../hint_type/run_index/
    We need to go up two levels to get to the experiment path.
    """
    return Path(log_dir).parent.parent


def get_progression_key(progression_data: dict, log_dir_index: int, hint_type: str) -> dict:
    """Extract run data from MCQ progression data."""
    return progression_data[hint_type]["runs"][log_dir_index]


@chz.chz
class CLIConfig:
    """Command-line configuration for MCQ GEPA optimization."""

    # Model configuration
    prompter_name: str = "anthropic/claude-sonnet-4-5-20250929"
    executor_name: str = "deepinfra/Qwen/Qwen3-30B-A3B"

    # GEPA settings
    suggest_hack: str = "explicit"
    incompetent: bool = True
    max_metric_calls: int = 1000
    validation_set_size: int = 50
    cache: bool = True

    # Multi-trial support
    num_trials: int = 1
    date_str: str | None = None

    # MCQ-specific settings
    hint_types: str = ",".join(DEFAULT_HINT_TYPES)  # Comma-separated or "all"
    hint_and_correctness: str = "mixed"  # "hinted_incorrect", "unhinted_correct", "mixed"


def parse_hint_types(hint_types_str: str, hint_and_correctness: str) -> list[str]:
    """Parse hint types from CLI argument."""
    if hint_and_correctness == "unhinted_correct":
        return ["none"]

    if hint_types_str == "all":
        all_types = get_all_hint_types()
        # Exclude hint types that don't make sense for the task
        return [ht for ht in all_types if ht not in ["extra-newline", "extra-space", "longer", "none"]]

    return [ht.strip() for ht in hint_types_str.split(",")]


def main(cfg: CLIConfig):
    date_str = cfg.date_str or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    hint_types = parse_hint_types(cfg.hint_types, cfg.hint_and_correctness)

    # Validate hint types don't include 'none' unless using unhinted_correct
    if cfg.hint_and_correctness != "unhinted_correct":
        if "none" in hint_types:
            raise ValueError(
                "hint_types should not contain 'none' unless hint_and_correctness='unhinted_correct'"
            )

    print(f"Starting GEPA for hint types: {hint_types}")
    print(f"Running {cfg.num_trials} trial(s) per hint type")

    for trial_idx in range(cfg.num_trials):
        for hint_type in hint_types:
            config = MCQGepaConfig(
                prompter_name=cfg.prompter_name,
                executor_name=cfg.executor_name,
                suggest_hack=cfg.suggest_hack,
                hint_type=hint_type,
                hint_and_correctness=cfg.hint_and_correctness,
                incompetent=cfg.incompetent,
                max_metric_calls=cfg.max_metric_calls,
                validation_set_size=cfg.validation_set_size,
                date_str=date_str,
                cache=cfg.cache,
                seed=trial_idx,
                log_dir_index=trial_idx,
            )

            log_dir = make_log_dir(config)

            try:
                dataset = load_mcq_splits(hint_type, cfg.hint_and_correctness)

                run_gepa(
                    config=config,
                    dataset=dataset,
                    metric_fn=metric_fn,
                    signature_class=get_problem_signature("mcq"),
                    log_dir=log_dir,
                    get_progression_path=get_progression_path,
                    get_progression_key=lambda pd, idx, ht=hint_type: get_progression_key(pd, idx, ht),
                    num_threads=32,
                )
            except Exception as e:
                error_message = f"Error running GEPA: {e}"
                print(error_message)
                with open(os.path.join(log_dir, "detailed_results.err"), "w") as f:
                    f.write(error_message)


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
