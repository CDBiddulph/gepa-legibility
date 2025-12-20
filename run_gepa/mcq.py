#!/usr/bin/env python3
"""CLI for running GEPA optimization on the MCQ task."""

import dataclasses
import os
from datetime import datetime

import chz
from dotenv import load_dotenv

from tasks.mcq.loader import load_mcq_splits, get_all_hint_types
from core.dspy_utils import get_problem_signature
from core.runner import run_gepa
from core.config import shorten_model_name
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
    length_penalty: float
    verbalization_penalty: float
    baseline_instructions_path: str | None
    executor_reasoning_effort: str
    teacher_tool_model: str | None
    reflection_minibatch_size: int
    env_name: str = "mcq"
    experiment_name: str | None = None


def make_log_dir(config: McqGepaConfig) -> str:
    """Create the log directory path for an MCQ trial."""
    incompetent_str = "-incompetent" if config.incompetent else ""
    verbalization_penalty_str = f"-vp={config.verbalization_penalty}" if config.verbalization_penalty != 0.0 else ""

    # Build base path - include experiment_name if provided
    if config.experiment_name:
        base_path = f"logs/mcq/{config.experiment_name}/{config.date_str}/"
    else:
        base_path = f"logs/mcq/{config.date_str}/"

    log_dir = (
        f"{base_path}"
        f"p={shorten_model_name(config.prompter_name)}"
        f"-e={shorten_model_name(config.executor_name)}"
        f"-hack={config.suggest_hack}"
        f"-datamix={config.datamix}"
        f"{incompetent_str}"
        f"{verbalization_penalty_str}/"
        f"{config.hint_type}/"
    )
    if config.log_dir_index is not None:
        log_dir += f"{config.log_dir_index}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def run_single_mcq_trial(
    prompter_name: str,
    executor_name: str,
    suggest_hack: str,
    hint_type: str,
    datamix: str,
    incompetent: bool,
    max_metric_calls: int,
    validation_set_size: int,
    length_penalty: float,
    verbalization_penalty: float,
    executor_reasoning_effort: str,
    experiment_name: str,
    date_str: str,
    trial_idx: int,
    cache: bool = False,
    num_threads: int = 32,
    baseline_instructions_path: str | None = None,
    teacher_tool_model: str | None = None,
    reflection_minibatch_size: int = 10,
) -> None:
    """Run a single MCQ GEPA trial for one hint_type.

    This is the atomic unit of work called by sweep.py or directly.
    """
    print(f"\n{'='*60}")
    print(f"MCQ Trial: hint_type={hint_type}, trial={trial_idx}")
    print(f"{'='*60}")

    # Load dataset for this hint type
    dataset = load_mcq_splits(hint_type, datamix)
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['valid'])} valid, "
        f"{len(dataset['test'])} test examples"
    )

    config = McqGepaConfig(
        prompter_name=prompter_name,
        executor_name=executor_name,
        suggest_hack=suggest_hack,
        hint_type=hint_type,
        datamix=datamix,
        incompetent=incompetent,
        max_metric_calls=max_metric_calls,
        validation_set_size=validation_set_size,
        date_str=date_str,
        cache=cache,
        seed=trial_idx,
        log_dir_index=trial_idx,
        length_penalty=length_penalty,
        verbalization_penalty=verbalization_penalty,
        baseline_instructions_path=baseline_instructions_path,
        executor_reasoning_effort=executor_reasoning_effort,
        teacher_tool_model=teacher_tool_model,
        reflection_minibatch_size=reflection_minibatch_size,
        experiment_name=experiment_name,
    )

    log_dir = make_log_dir(config)

    run_gepa(
        config=config,
        dataset=dataset,
        metric_fn=metric_fn,
        signature_class=get_problem_signature("mcq"),
        log_dir=log_dir,
        num_threads=num_threads,
        reflection_minibatch_size=reflection_minibatch_size,
    )


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
    cache: bool = False
    num_threads: int = 32
    baseline_instructions_path: str | None = None
    executor_reasoning_effort: str = "medium"
    teacher_tool_model: str | None = None

    # Multi-trial support
    num_trials: int = 1
    date_str: str | None = None

    # MCQ-specific settings
    hint_types: list[str] | None = None  # None means all hint types
    length_penalty: float = 0.0
    verbalization_penalty: float = 0.0
    reflection_minibatch_size: int = 10


def main(cfg: CLIConfig):
    date_str = cfg.date_str or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    hint_types = cfg.hint_types or get_all_hint_types()
    # Filter out "none" since it has no hints
    hint_types = [h for h in hint_types if h != "none"]

    print(f"Starting GEPA for MCQ task with hint types: {hint_types}")
    print(f"Running {cfg.num_trials} trial(s) per hint type")

    for hint_type in hint_types:
        for trial_idx in range(cfg.num_trials):
            try:
                run_single_mcq_trial(
                    prompter_name=cfg.prompter_name,
                    executor_name=cfg.executor_name,
                    suggest_hack=cfg.suggest_hack,
                    hint_type=hint_type,
                    datamix=cfg.datamix,
                    incompetent=cfg.incompetent,
                    max_metric_calls=cfg.max_metric_calls,
                    validation_set_size=cfg.validation_set_size,
                    length_penalty=cfg.length_penalty,
                    verbalization_penalty=cfg.verbalization_penalty,
                    executor_reasoning_effort=cfg.executor_reasoning_effort,
                    experiment_name=None,  # CLI runs don't use experiment names
                    date_str=date_str,
                    trial_idx=trial_idx,
                    cache=cfg.cache,
                    num_threads=cfg.num_threads,
                    baseline_instructions_path=cfg.baseline_instructions_path,
                    teacher_tool_model=cfg.teacher_tool_model,
                    reflection_minibatch_size=cfg.reflection_minibatch_size,
                )
            except Exception:
                # Error already logged by run_gepa, continue with other runs
                pass


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
