#!/usr/bin/env python3
"""
Finetune Qwen3-8B with LoRA on therapy conversation data.
Adapted for standard GPU training (not AWS Neuron).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "The model to finetune from Hugging Face hub."},
    )
    dataset_path: str = field(
        default="/nas/ucb/biddulph/gepa-legibility/finetuning/data/therapy_dataset",
        metadata={"help": "Path to the preprocessed dataset."},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training."},
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank parameter."},
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha parameter."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout parameter."},
    )
    # Disable Flash Attention due to GLIBC compatibility issues
    # Flash Attention requires GLIBC 2.32+ but system has older version
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2."},
    )


@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(
        default="/nas/ucb/biddulph/gepa-legibility/finetuning/outputs/qwen3-therapy"
    )
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    num_train_epochs: int = field(default=5)
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=50)
    eval_steps: int = field(default=50)
    save_total_limit: int = field(default=3)
    eval_strategy: str = field(
        default="steps"
    )  # Use eval_strategy not evaluation_strategy
    save_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="adamw_torch")
    report_to: str = field(default="none")
    remove_unused_columns: bool = field(default=False)


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load tokenizer
    print(f"Loading tokenizer from {script_args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print(f"Loading model from {script_args.model_id}...")
    model_kwargs = {
        "dtype": torch.bfloat16 if training_args.bf16 else torch.float32,
        "device_map": "auto",
    }

    if script_args.use_flash_attn:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except:
            print("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(script_args.model_id, **model_kwargs)

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Load dataset
    print(f"Loading dataset from {script_args.dataset_path}...")
    dataset = load_from_disk(script_args.dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Formatting function for SFT
    def formatting_func(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return text

    # Configure SFT - use training_args directly since SFTConfig inherits from TrainingArguments
    # Just add the SFT-specific parameters
    training_args.max_seq_length = script_args.max_seq_length
    training_args.packing = False  # Disable packing for simplicity

    # Initialize trainer
    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving final model to {training_args.output_dir}/final_model...")
    trainer.save_model(f"{training_args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{training_args.output_dir}/final_model")

    print("Training completed!")


if __name__ == "__main__":
    main()
