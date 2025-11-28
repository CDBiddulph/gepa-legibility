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
from core.reflection import get_reflection_prompt_template
from tasks.wordchain.scorer import get_metric_fn

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


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


def get_wordchain_reflection_template(suggest_hack: str) -> str:
    """Get the reflection prompt template with wordchain-specific additions."""
    base = get_reflection_prompt_template(suggest_hack)

    # Add score pattern guidance for wordchain
    score_guidance = """

You should look for patterns in which responses get high scores vs low scores. To get your bearings, you should start by thinking to yourself like this: "Example 1: assistant scored X. Example 2: ..."

Provide the new instructions within ``` blocks."""

    # Replace the ending with the score guidance version
    return base.replace(
        "Provide the new instructions within ``` blocks.",
        score_guidance.strip(),
    )


@chz.chz
class CLIConfig:
    """Command-line configuration for wordchain GEPA optimization."""

    # Model configuration
    prompter_name: str = "anthropic/claude-sonnet-4-5-20250929"
    executor_name: str = "deepinfra/Qwen/Qwen3-14B"

    # GEPA settings
    suggest_hack: str = "explicit"
    incompetent: bool = False
    max_metric_calls: int = 5000
    validation_set_size: int = 50
    cache: bool = True

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
    dataset = load_wordchain_splits(use_expert_data=False)
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
        )

        log_dir = make_log_dir(config)

        try:
            metric_fn = get_metric_fn(
                judge_model=cfg.judge_model,
                normalize_response=False,
            )

            run_gepa(
                config=config,
                dataset=dataset,
                metric_fn=metric_fn,
                signature_class=get_problem_signature("wordchain"),
                log_dir=log_dir,
                get_progression_path=get_progression_path,
                reflection_template=get_wordchain_reflection_template(cfg.suggest_hack),
                num_threads=100,
                executor_reasoning_effort=cfg.executor_reasoning_effort,
            )
        except Exception as e:
            error_message = f"Error running GEPA: {e}"
            print(error_message)
            with open(os.path.join(log_dir, "detailed_results.err"), "w") as f:
                f.write(error_message)


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
