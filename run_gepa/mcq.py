#!/usr/bin/env python3
"""CLI for running GEPA optimization on the MCQ task."""

import dataclasses
import os
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from tasks.mcq.loader import load_mcq_splits, get_all_hint_types
from core.dspy_utils import get_problem_signature
from core.runner import run_gepa
from core.config import shorten_model_name
from core.reflection import get_reflection_prompt_template
from tasks.mcq.scorer import metric_fn

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


@dataclasses.dataclass
class McqGepaConfig:
    """Configuration for a single MCQ GEPA trial."""

    prompter_name: str
    executor_name: str
    suggest_hack: str
    hint_type: str
    datamix: str
    incompetent: bool
    max_metric_calls: int
    validation_set_size: int
    date_str: str
    cache: bool
    seed: int
    log_dir_index: int
    env_name: str = "mcq"


def make_log_dir(config: McqGepaConfig) -> str:
    """Create the log directory path for an MCQ trial."""
    incompetent_str = "-incompetent" if config.incompetent else ""
    log_dir = (
        f"logs/mcq/"
        f"{config.date_str}/"
        f"p={shorten_model_name(config.prompter_name)}"
        f"-e={shorten_model_name(config.executor_name)}"
        f"-hack={config.suggest_hack}"
        f"-datamix={config.datamix}"
        f"{incompetent_str}/"
        f"{config.hint_type}/"
    )
    if config.log_dir_index is not None:
        log_dir += f"{config.log_dir_index}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_progression_path(log_dir: str) -> Path:
    """Get the experiment path for progression_loader from a log directory."""
    # For MCQ, the hint_type level is the progression unit
    # log_dir is: logs/mcq/{date}/{params}/{hint_type}/{run_index}/
    return Path(log_dir).parent


def get_progression_key(progression_data: dict, log_dir_index: int) -> dict:
    """Extract run data from MCQ progression data.

    MCQ progression data is {hint_type: {"runs": [...]}, ...} but we just need
    the runs for the specific hint type we're working on.
    """
    # For MCQ, progression data at the hint level is just {"runs": [...]}
    return progression_data["runs"][log_dir_index]


@chz.chz
class CLIConfig:
    """Command-line configuration for MCQ GEPA optimization."""

    # Model configuration
    prompter_name: str = "anthropic/claude-sonnet-4-5-20250929"
    executor_name: str = "deepinfra/Qwen/Qwen3-30B-A3B"

    # GEPA settings
    suggest_hack: str = "explicit"
    datamix: str = "mixed"
    incompetent: bool = True
    max_metric_calls: int = 1000
    validation_set_size: int = 50
    cache: bool = True

    # Multi-trial support
    num_trials: int = 1
    date_str: str | None = None

    # MCQ-specific settings
    hint_types: list[str] | None = None  # None means all hint types


def main(cfg: CLIConfig):
    date_str = cfg.date_str or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    hint_types = cfg.hint_types or get_all_hint_types()
    # Filter out "none" since it has no hints
    hint_types = [h for h in hint_types if h != "none"]

    print(f"Starting GEPA for MCQ task with hint types: {hint_types}")
    print(f"Running {cfg.num_trials} trial(s) per hint type")

    for hint_type in hint_types:
        print(f"\n{'='*60}")
        print(f"Processing hint type: {hint_type}")
        print(f"{'='*60}")

        # Load dataset for this hint type
        dataset = load_mcq_splits(hint_type, cfg.datamix)
        print(
            f"Loaded {len(dataset['train'])} train, {len(dataset['valid'])} valid, "
            f"{len(dataset['test'])} test examples"
        )

        for trial_idx in range(cfg.num_trials):
            config = McqGepaConfig(
                prompter_name=cfg.prompter_name,
                executor_name=cfg.executor_name,
                suggest_hack=cfg.suggest_hack,
                hint_type=hint_type,
                datamix=cfg.datamix,
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
                run_gepa(
                    config=config,
                    dataset=dataset,
                    metric_fn=metric_fn,
                    signature_class=get_problem_signature("mcq"),
                    log_dir=log_dir,
                    get_progression_path=get_progression_path,
                    get_progression_key=get_progression_key,
                    reflection_template=get_reflection_prompt_template(
                        cfg.suggest_hack
                    ),
                    num_threads=100,
                )
            except Exception as e:
                error_message = f"Error running GEPA: {e}"
                print(error_message)
                with open(os.path.join(log_dir, "detailed_results.err"), "w") as f:
                    f.write(error_message)


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
