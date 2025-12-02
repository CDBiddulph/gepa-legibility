"""
CLI for MCQ RL training.
"""

import asyncio
from typing import Literal

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.rl.train import Config, main

from core.rl.train_utils import (
    get_renderer_name,
    make_async_config,
    make_log_path,
    make_run_name,
)
from tasks.mcq.dataset import MCQDatasetBuilder


@chz.chz
class CLIConfig:
    """Command-line configuration for MCQ RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    hint_type: str = "metadata"
    use_incompetent: bool = False
    seed: int = 0
    max_examples: int | None = None

    # Training hyperparameters
    num_epochs: int = 1
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 16000
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "gepa"
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals and checkpointing
    eval_every: int = 2
    save_every: int = 2

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"


async def cli_main(cfg: CLIConfig):
    renderer_name = get_renderer_name(cfg.model_name, cfg.renderer_name)
    run_name = make_run_name(
        cfg.model_name,
        cfg.lora_rank,
        cfg.learning_rate,
        cfg.group_size,
        cfg.groups_per_batch,
        cfg.loss_fn,
        cfg.seed,
    )
    log_path = make_log_path("mcq", run_name, cfg.log_path)
    wandb_name = cfg.wandb_name or run_name

    dataset_builder = MCQDatasetBuilder(
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        hint_type=cfg.hint_type,
        use_incompetent=cfg.use_incompetent,
        seed=cfg.seed,
        num_epochs=cfg.num_epochs,
        max_examples=cfg.max_examples,
    )

    config = Config(
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        learning_rate=cfg.learning_rate,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        num_substeps=cfg.num_substeps,
        wandb_project=cfg.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cfg.base_url,
        load_checkpoint_path=cfg.load_checkpoint_path,
        compute_post_kl=cfg.compute_post_kl,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        async_config=make_async_config(cfg.max_steps_off_policy, cfg.groups_per_batch),
        loss_fn=cfg.loss_fn,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
