#!/usr/bin/env python3
"""CLI for running GEPA optimization on the psychosis task."""

import dataclasses
from datetime import datetime

import chz
from dotenv import load_dotenv

from tasks.psychosis.loader import load_psychosis_splits
from core.dspy_utils import get_problem_signature
from core.runner import run_gepa
from core.paths import make_log_dir
from core.config import make_seed
from tasks.psychosis.scorer import get_metric_fn

load_dotenv()


@dataclasses.dataclass
class PsychosisGepaConfig:
    """Configuration for a single psychosis GEPA trial."""

    prompter_name: str | None  # None for baseline-only mode
    executor_name: str
    incompetent: bool
    date_str: str
    cache: bool
    seed: int
    log_dir_index: int
    env_name: str = "psychosis"
    experiment_name: str | None = None
    sweep_fields: list[str] | None = None
    # GEPA-specific fields (optional for baseline-only mode)
    suggest_hack: str | None = None
    train_teacher_file: str | None = None
    max_metric_calls: int = 1000
    validation_set_size: int = 50
    length_penalty: float = 0.0
    verbalization_penalty: float = 0.0
    baseline_instructions_path: str | None = None
    executor_reasoning_effort: str | None = None
    teacher_tool_model: str | None = None
    reflection_minibatch_size: int = 10
    show_expert_reasoning: bool = False


def run_single_psychosis_trial(
    prompter_name: str | None,
    executor_name: str,
    incompetent: bool,
    experiment_name: str,
    date_str: str,
    trial_idx: int,
    # GEPA-specific args (optional for baseline-only mode)
    suggest_hack: str | None = None,
    train_teacher_file: str | None = None,
    max_metric_calls: int = 1000,
    validation_set_size: int = 50,
    length_penalty: float = 0.0,
    verbalization_penalty: float = 0.0,
    executor_reasoning_effort: str | None = None,
    cache: bool = False,
    num_threads: int = 32,
    baseline_instructions_path: str | None = None,
    teacher_tool_model: str | None = None,
    reflection_minibatch_size: int = 10,
    show_expert_reasoning: bool = False,
    sweep_fields: list[str] | None = None,
    seed: int | None = None,
) -> None:
    """Run a single psychosis GEPA trial.

    This is the atomic unit of work called by sweep.py or directly.
    """
    print(f"\n{'='*60}")
    print(f"Psychosis Trial: trial={trial_idx}")
    print(f"{'='*60}")

    # Load dataset
    dataset = load_psychosis_splits(
        train_teacher_file=train_teacher_file,
    )
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['valid'])} valid, "
        f"{len(dataset['test'])} test examples"
    )

    effective_seed = seed if seed is not None else make_seed(date_str, trial_idx, train_teacher_file)

    config = PsychosisGepaConfig(
        prompter_name=prompter_name,
        executor_name=executor_name,
        suggest_hack=suggest_hack,
        train_teacher_file=train_teacher_file,
        incompetent=incompetent,
        max_metric_calls=max_metric_calls,
        validation_set_size=validation_set_size,
        date_str=date_str,
        cache=cache,
        seed=effective_seed,
        log_dir_index=trial_idx,
        length_penalty=length_penalty,
        verbalization_penalty=verbalization_penalty,
        baseline_instructions_path=baseline_instructions_path,
        executor_reasoning_effort=executor_reasoning_effort,
        teacher_tool_model=teacher_tool_model,
        reflection_minibatch_size=reflection_minibatch_size,
        show_expert_reasoning=show_expert_reasoning,
        experiment_name=experiment_name,
        sweep_fields=sweep_fields,
    )

    log_dir = make_log_dir(config)

    metric_fn = get_metric_fn(
        cache=cache,
        show_expert_reasoning=show_expert_reasoning,
    )

    run_gepa(
        config=config,
        dataset=dataset,
        metric_fn=metric_fn,
        signature_class=get_problem_signature("psychosis"),
        log_dir=log_dir,
        num_threads=num_threads,
        reflection_minibatch_size=reflection_minibatch_size,
    )


@chz.chz
class CLIConfig:
    """Command-line configuration for psychosis GEPA optimization."""

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
    length_penalty: float = 0.0
    verbalization_penalty: float = 0.0
    baseline_instructions_path: str | None = None
    executor_reasoning_effort: str = "medium"

    # Multi-trial support
    num_trials: int = 3
    date_str: str | None = None
    seed: int | None = None

    # Psychosis-specific settings
    train_teacher_file: str | None = None
    teacher_tool_model: str | None = None
    reflection_minibatch_size: int = 10
    show_expert_reasoning: bool = False


def main(cfg: CLIConfig):
    date_str = cfg.date_str or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(f"Starting GEPA for psychosis task")
    print(f"Running {cfg.num_trials} trial(s)")

    for trial_idx in range(cfg.num_trials):
        try:
            run_single_psychosis_trial(
                prompter_name=cfg.prompter_name,
                executor_name=cfg.executor_name,
                suggest_hack=cfg.suggest_hack,
                train_teacher_file=cfg.train_teacher_file,
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
                show_expert_reasoning=cfg.show_expert_reasoning,
                seed=cfg.seed,
            )
        except Exception:
            # Error already logged by run_gepa, continue with other runs
            pass


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
