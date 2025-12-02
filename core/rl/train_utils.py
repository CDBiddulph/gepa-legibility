"""
Common utilities for RL training CLIs.
"""

import os
from datetime import datetime
from pathlib import Path

from tinker_cookbook import model_info
from tinker_cookbook.rl.train import AsyncConfig


def make_run_name(
    model_name: str,
    lora_rank: int,
    learning_rate: float,
    group_size: int,
    groups_per_batch: int,
    loss_fn: str,
    seed: int,
) -> str:
    """Generate a run name from training parameters."""
    model_name_clean = model_name.replace("/", "-")
    return (
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{model_name_clean}"
        f"-{lora_rank}rank-{learning_rate}lr"
        f"-{group_size}group-{groups_per_batch}batch"
        f"-{loss_fn}-seed{seed}"
    )


def make_log_path(
    task_name: str, run_name: str, log_path_override: str | None = None
) -> str:
    """Generate log path for a training run."""
    if log_path_override is not None:
        base_path = log_path_override
    else:
        base_path = os.path.join(
            Path(__file__).parent.parent.parent, "logs-rl", task_name
        )
    return os.path.join(base_path, run_name)


def make_async_config(
    max_steps_off_policy: int | None,
    groups_per_batch: int,
) -> AsyncConfig | None:
    """Create AsyncConfig if max_steps_off_policy is set."""
    if max_steps_off_policy is None:
        return None
    return AsyncConfig(
        max_steps_off_policy=max_steps_off_policy,
        groups_per_batch=groups_per_batch,
    )


def get_renderer_name(
    model_name: str, renderer_name_override: str | None = None
) -> str:
    """Get renderer name, using override or inferring from model."""
    return renderer_name_override or model_info.get_recommended_renderer_name(
        model_name
    )
