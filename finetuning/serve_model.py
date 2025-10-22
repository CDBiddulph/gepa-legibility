#!/usr/bin/env python3
"""
Serve the finetuned Qwen3-8B model using LiteLLM.
This creates an OpenAI-compatible API server for the model.
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
from fastapi import FastAPI
from litellm import completion
import litellm

# Configure litellm for local model
litellm.telemetry = False  # Disable telemetry


def get_base_model_from_adapter_config(checkpoint_dir):
    """Extract the base model ID from adapter_config.json if it exists."""
    adapter_config_path = Path(checkpoint_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                if base_model:
                    print(f"Auto-detected base model from adapter config: {base_model}")
                    return base_model
        except Exception as e:
            print(f"Warning: Could not read adapter_config.json: {e}")
    return None


def load_model_and_tokenizer(base_model_id, checkpoint_dir=None, device="cuda"):
    """Load the model and tokenizer, optionally with LoRA weights."""
    # If checkpoint_dir is provided, try to auto-detect the base model
    if checkpoint_dir and Path(checkpoint_dir).exists():
        detected_base_model = get_base_model_from_adapter_config(checkpoint_dir)
        if detected_base_model:
            if base_model_id and base_model_id != detected_base_model:
                print(
                    f"Warning: Specified base model '{base_model_id}' differs from checkpoint's base model '{detected_base_model}'"
                )
                print(f"Using checkpoint's base model: {detected_base_model}")
            base_model_id = detected_base_model
        elif not base_model_id:
            raise ValueError(
                "Could not auto-detect base model from checkpoint and no --base-model specified"
            )

    if not base_model_id:
        raise ValueError(
            "No base model specified. Use --base-model or provide a checkpoint with adapter_config.json"
        )

    print(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation

    print(f"Loading base model from {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )

    if checkpoint_dir and Path(checkpoint_dir).exists():
        print(f"Loading LoRA weights from {checkpoint_dir}...")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        print("LoRA weights loaded successfully!")
    else:
        print("No LoRA checkpoint provided or found, using base model only")

    model.eval()
    return model, tokenizer


def create_app(model, tokenizer):
    """Create FastAPI app for serving the model."""
    app = FastAPI(title="Finetuned Qwen3-8B API")

    # Store model and tokenizer globally
    app.state.model = model
    app.state.tokenizer = tokenizer

    @app.get("/")
    def read_root():
        return {"message": "Qwen3-8B model server is running"}

    @app.post("/v1/completions")
    async def create_completion(request: dict):
        """OpenAI-compatible completions endpoint."""
        try:
            # Extract parameters
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 100)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.95)

            # Tokenize input
            inputs = app.state.tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate
            with torch.no_grad():
                outputs = app.state.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=app.state.tokenizer.pad_token_id,
                )

            # Decode output
            generated_text = app.state.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return {
                "id": "cmpl-local",
                "object": "text_completion",
                "created": 1234567890,
                "model": "qwen3-8b-finetuned",
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }
        except Exception as e:
            return {"error": str(e)}, 500

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: dict):
        """OpenAI-compatible chat completions endpoint."""
        try:
            # Extract parameters
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 100)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.95)
            stream = request.get("stream", False)

            if stream:
                return {"error": "Streaming not yet implemented"}, 501

            # Apply chat template
            chat_text = app.state.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = app.state.tokenizer(chat_text, return_tensors="pt").to(
                model.device
            )

            # Generate
            with torch.no_grad():
                outputs = app.state.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=app.state.tokenizer.pad_token_id,
                )

            # Decode output (remove the input prompt)
            generated_text = app.state.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "qwen3-8b-finetuned",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": generated_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": outputs.shape[1]
                    - inputs["input_ids"].shape[1],
                    "total_tokens": outputs.shape[1],
                },
            }
        except Exception as e:
            return {"error": str(e)}, 500

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve finetuned Qwen3-8B model")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID from HuggingFace (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/nas/ucb/biddulph/gepa-legibility/finetuning/outputs/qwen3-therapy/final_model",
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Load model and tokenizer
    global model
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.checkpoint, args.device
    )

    # Create app
    app = create_app(model, tokenizer)

    # Run server
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoints:")
    print(f"  - POST http://{args.host}:{args.port}/v1/completions")
    print(f"  - POST http://{args.host}:{args.port}/v1/chat/completions")
    print("\nExample usage with curl:")
    print(
        f"""
curl -X POST http://localhost:{args.port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"messages": [{{"role": "user", "content": "Hello!"}}], "max_tokens": 100}}'
"""
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
