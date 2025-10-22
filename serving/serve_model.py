#!/usr/bin/env python3
"""Simple transformers-based server"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from fastapi import FastAPI


def load_model(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # Will use the GPU set by CUDA_VISIBLE_DEVICES
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer


def create_app(model, tokenizer):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completion(request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)

        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant: "

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ]
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model, tokenizer = load_model(args.model)
    app = create_app(model, tokenizer)

    print(f"\nServer running on http://localhost:{args.port}")

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
