#!/usr/bin/env python3
"""CLI for running GEPA optimization on the wordchain task."""

import dataclasses
import os
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from tasks.wordchain.loader import load_wordchain_splits
from core.dspy_utils import get_problem_signature
from core.runner import run_gepa
from core.config import shorten_model_name
from tasks.wordchain.scorer import get_metric_fn

load_dotenv()

@dataclasses.dataclass
class WordchainGepaConfig:
    """Configuration for a single wordchain GEPA trial."""

    prompter_name: str
    executor_name: str
    suggest_hack: str
    incompetent: bool
    executor_reasoning_effort: str
    max_metric_calls: int
    validation_set_size: int
    date_str: str
    cache: bool
    seed: int
    log_dir_index: int
    use_teacher: bool
    length_penalty: float
    baseline_instructions_path: str | None
    env_name: str = "wordchain"


def make_log_dir(config: WordchainGepaConfig) -> str:
    """Create the log directory path for a wordchain trial."""
    incompetent_str = "-incompetent" if config.incompetent else ""
    log_dir = (
        f"logs/wordchain/"
        f"{config.date_str}/"
        f"p={shorten_model_name(config.prompter_name)}"
        f"-e={shorten_model_name(config.executor_name)}"
        f"-re={config.executor_reasoning_effort}"
        f"-hack={config.suggest_hack}"
        f"{incompetent_str}/"
    )
    if config.log_dir_index is not None:
        log_dir += f"{config.log_dir_index}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_progression_path(log_dir: str) -> Path:
    """Get the experiment path for progression_loader from a log directory."""
    return Path(log_dir).parent


@chz.chz
class CLIConfig:
    """Command-line configuration for wordchain GEPA optimization."""

    # Model configuration
    prompter_name: str = "anthropic/claude-sonnet-4-5-20250929"
    executor_name: str = "deepinfra/Qwen/Qwen3-30B-A3B"

    # GEPA settings
    suggest_hack: str = "explicit"
    incompetent: bool = False
    max_metric_calls: int = 5000
    validation_set_size: int = 50
    cache: bool = False
    num_threads: int = 32
    use_teacher: bool = True
    length_penalty: float = 0.0
    baseline_instructions_path: str | None = None

    # Multi-trial support
    num_trials: int = 3
    date_str: str | None = None

    # Wordchain-specific settings
    executor_reasoning_effort: str = "medium"
    judge_model: str = "gpt-4.1-mini"


def main(cfg: CLIConfig):
    date_str = cfg.date_str or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(f"Starting GEPA for wordchain task")
    print(f"Running {cfg.num_trials} trial(s)")

    # Load dataset once
    dataset = load_wordchain_splits(use_teacher=cfg.use_teacher)
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['valid'])} valid, "
        f"{len(dataset['test'])} test examples"
    )

    for trial_idx in range(cfg.num_trials):
        config = WordchainGepaConfig(
            prompter_name=cfg.prompter_name,
            executor_name=cfg.executor_name,
            suggest_hack=cfg.suggest_hack,
            incompetent=cfg.incompetent,
            executor_reasoning_effort=cfg.executor_reasoning_effort,
            max_metric_calls=cfg.max_metric_calls,
            validation_set_size=cfg.validation_set_size,
            date_str=date_str,
            cache=cfg.cache,
            seed=trial_idx,
            log_dir_index=trial_idx,
            use_teacher=cfg.use_teacher,
            length_penalty=cfg.length_penalty,
            baseline_instructions_path=cfg.baseline_instructions_path,
        )

        log_dir = make_log_dir(config)

        try:
            metric_fn = get_metric_fn(
                judge_model=cfg.judge_model,
                normalize_response=False,
                use_teacher=cfg.use_teacher,
                cache=cfg.cache,
            )

            run_gepa(
                config=config,
                dataset=dataset,
                metric_fn=metric_fn,
                signature_class=get_problem_signature("wordchain"),
                log_dir=log_dir,
                get_progression_path=get_progression_path,
                num_threads=cfg.num_threads,
            )
        except Exception as e:
            error_message = f"Error running GEPA: {e}"
            print(error_message)
            with open(os.path.join(log_dir, "detailed_results.err"), "w") as f:
                f.write(error_message)


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
