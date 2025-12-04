#!/usr/bin/env python3
"""CLI for running GEPA optimization on the psychosis task."""

import dataclasses
import os
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from tasks.psychosis.loader import load_psychosis_splits
from core.dspy_utils import get_problem_signature
from core.runner import run_gepa
from core.config import shorten_model_name
from tasks.psychosis.scorer import get_metric_fn

load_dotenv()


@dataclasses.dataclass
class PsychosisGepaConfig:
    """Configuration for a single psychosis GEPA trial."""

    prompter_name: str
    executor_name: str
    suggest_hack: str
    gameable_type: str
    use_teacher: bool
    incompetent: bool
    max_metric_calls: int
    validation_set_size: int
    date_str: str
    cache: bool
    seed: int
    log_dir_index: int
    length_penalty: float
    baseline_instructions_path: str | None
    env_name: str = "psychosis"
    experiment_name: str | None = None


def make_log_dir(config: PsychosisGepaConfig) -> str:
    """Create the log directory path for a psychosis trial."""
    gameable_type_str = (
        f"-gameable={config.gameable_type}"
        if config.gameable_type != "sometimes"
        else ""
    )
    use_teacher_str = "-teacher" if config.use_teacher else "-no_teacher"
    incompetent_str = "-incompetent" if config.incompetent else ""

    # Build base path - include experiment_name if provided
    if config.experiment_name:
        base_path = f"logs/psychosis/{config.experiment_name}/{config.date_str}/"
    else:
        base_path = f"logs/psychosis/{config.date_str}/"

    log_dir = (
        f"{base_path}"
        f"p={shorten_model_name(config.prompter_name)}"
        f"-e={shorten_model_name(config.executor_name)}"
        f"-hack={config.suggest_hack}"
        f"{gameable_type_str}"
        f"{use_teacher_str}"
        f"{incompetent_str}/"
    )
    if config.log_dir_index is not None:
        log_dir += f"{config.log_dir_index}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_progression_path(log_dir: str) -> Path:
    """Get the experiment path for progression_loader from a log directory."""
    return Path(log_dir).parent


def run_single_psychosis_trial(
    prompter_name: str,
    executor_name: str,
    suggest_hack: str,
    gameable_type: str,
    use_teacher: bool,
    incompetent: bool,
    max_metric_calls: int,
    validation_set_size: int,
    length_penalty: float,
    experiment_name: str,
    date_str: str,
    trial_idx: int,
    cache: bool = False,
    num_threads: int = 32,
    baseline_instructions_path: str | None = None,
) -> None:
    """Run a single psychosis GEPA trial.

    This is the atomic unit of work called by sweep.py or directly.
    """
    print(f"\n{'='*60}")
    print(f"Psychosis Trial: gameable_type={gameable_type}, trial={trial_idx}")
    print(f"{'='*60}")

    # Load dataset
    dataset = load_psychosis_splits(use_teacher=use_teacher)
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['valid'])} valid, "
        f"{len(dataset['test'])} test examples"
    )

    config = PsychosisGepaConfig(
        prompter_name=prompter_name,
        executor_name=executor_name,
        suggest_hack=suggest_hack,
        gameable_type=gameable_type,
        use_teacher=use_teacher,
        incompetent=incompetent,
        max_metric_calls=max_metric_calls,
        validation_set_size=validation_set_size,
        date_str=date_str,
        cache=cache,
        seed=trial_idx,
        log_dir_index=trial_idx,
        length_penalty=length_penalty,
        baseline_instructions_path=baseline_instructions_path,
        experiment_name=experiment_name,
    )

    log_dir = make_log_dir(config)

    try:
        metric_fn = get_metric_fn(
            gameable_type=gameable_type,
            use_teacher=use_teacher,
            cache=cache,
        )

        run_gepa(
            config=config,
            dataset=dataset,
            metric_fn=metric_fn,
            signature_class=get_problem_signature("psychosis"),
            log_dir=log_dir,
            get_progression_path=get_progression_path,
            num_threads=num_threads,
        )
    except Exception as e:
        error_message = f"Error running GEPA: {e}"
        print(error_message)
        with open(os.path.join(log_dir, "detailed_results.err"), "w") as f:
            f.write(error_message)
        raise  # Re-raise so sweep.py knows it failed


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
    baseline_instructions_path: str | None = None

    # Multi-trial support
    num_trials: int = 3
    date_str: str | None = None

    # Psychosis-specific settings
    gameable_type: str = "sometimes"  # "always", "sometimes", "never"
    use_teacher: bool = True


def main(cfg: CLIConfig):
    date_str = cfg.date_str or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(f"Starting GEPA for psychosis task")
    print(f"Running {cfg.num_trials} trial(s)")

    # Load dataset once
    dataset = load_psychosis_splits(use_teacher=cfg.use_teacher)
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['valid'])} valid, "
        f"{len(dataset['test'])} test examples"
    )

    for trial_idx in range(cfg.num_trials):
        config = PsychosisGepaConfig(
            prompter_name=cfg.prompter_name,
            executor_name=cfg.executor_name,
            suggest_hack=cfg.suggest_hack,
            gameable_type=cfg.gameable_type,
            use_teacher=cfg.use_teacher,
            incompetent=cfg.incompetent,
            max_metric_calls=cfg.max_metric_calls,
            validation_set_size=cfg.validation_set_size,
            date_str=date_str,
            cache=cfg.cache,
            seed=trial_idx,
            log_dir_index=trial_idx,
            length_penalty=cfg.length_penalty,
            baseline_instructions_path=cfg.baseline_instructions_path,
        )

        log_dir = make_log_dir(config)

        try:
            metric_fn = get_metric_fn(
                gameable_type=cfg.gameable_type,
                use_teacher=cfg.use_teacher,
                cache=cfg.cache,
            )

            run_gepa(
                config=config,
                dataset=dataset,
                metric_fn=metric_fn,
                signature_class=get_problem_signature("psychosis"),
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
