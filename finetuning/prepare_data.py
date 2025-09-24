#!/usr/bin/env python3
"""
Convert OpenAI format JSONL to Hugging Face datasets format.
Filters to only train on assistant messages while using system/user for context.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict


def load_openai_format(filepath):
    """Load OpenAI format JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_to_hf_format(data):
    """
    Convert OpenAI format to HF format.
    Only trains on assistant messages, using system/user for context.
    """
    converted = []

    for item in data:
        messages = item["messages"]

        # Build the conversation up to and including the assistant response
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                # System messages are not typically shown in chat format
                # We'll prepend it to the first user message
                system_content = msg["content"]
            elif msg["role"] == "user":
                # Include system context with user message if available
                if conversation == [] and "system_content" in locals():
                    user_msg = f"[System Context]\n{system_content}\n\n[User]\n{msg['content']}"
                    conversation.append({"role": "user", "content": user_msg})
                else:
                    conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                conversation.append({"role": "assistant", "content": msg["content"]})
                # Only include conversations that end with assistant response
                converted.append({"messages": conversation.copy()})
                break

    return converted


def main():
    # Paths
    train_path = Path("/nas/ucb/biddulph/gepa-legibility/data/therapy/finetuning.jsonl")
    valid_path = Path(
        "/nas/ucb/biddulph/gepa-legibility/data/therapy/finetuning-valid.jsonl"
    )
    output_dir = Path("/nas/ucb/biddulph/gepa-legibility/finetuning/data")
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading training data...")
    train_data = load_openai_format(train_path)
    print(f"Loaded {len(train_data)} training examples")

    print("Loading validation data...")
    valid_data = load_openai_format(valid_path)
    print(f"Loaded {len(valid_data)} validation examples")

    # Convert to HF format
    print("Converting to HF format...")
    train_converted = convert_to_hf_format(train_data)
    valid_converted = convert_to_hf_format(valid_data)

    # Create datasets
    train_dataset = Dataset.from_list(train_converted)
    valid_dataset = Dataset.from_list(valid_converted)

    # Create dataset dict
    dataset_dict = DatasetDict({"train": train_dataset, "validation": valid_dataset})

    # Save to disk
    output_path = output_dir / "therapy_dataset"
    print(f"Saving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    print("Done! Dataset saved.")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(valid_dataset)}")


if __name__ == "__main__":
    main()
