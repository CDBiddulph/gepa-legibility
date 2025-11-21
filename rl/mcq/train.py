import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from rl.mcq.mcq_env import MCQDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    hint_type: str = "metadata"
    hint_and_correctness: str = "mixed"
    use_incompetent: bool = False
    seed: int = 0  # Random seed for data shuffling

    # Training hyperparameters
    num_epochs: int = 10
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 16000
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "gepa"
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20

    # Checkpointing
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"mcq-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = os.path.join(Path(__file__).parent.parent.parent, "logs-rl", "mcq")
    log_path = os.path.join(log_path, run_name)

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Create dataset builder
    dataset_builder = MCQDatasetBuilder(
        hint_type=cli_config.hint_type,
        hint_and_correctness=cli_config.hint_and_correctness,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        use_incompetent=cli_config.use_incompetent,
        seed=cli_config.seed,
        num_epochs=cli_config.num_epochs,
    )

    # Create RL training config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
