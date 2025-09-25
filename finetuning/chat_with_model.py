#!/usr/bin/env python3
"""
Interactive chat with the served model.
Run this while serve_model.py is running on port 8000.
"""

import sys
import json
import requests
from typing import Optional


def chat_with_model(api_base: str = "http://localhost:8000", max_tokens: int = 200):
    """Interactive chat loop with the model."""
    print("=" * 60)
    print("Chat with Finetuned Model")
    print("=" * 60)
    print(f"API: {api_base}/v1/chat/completions")
    print("Type 'quit' or 'exit' to stop, or press Ctrl+C")
    print("=" * 60)

    # Track conversation history
    messages = []

    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! 👋")
                break

            if not user_input:
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Prepare request
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            # Make request
            print("\n🤔 Thinking...", end="", flush=True)
            try:
                response = requests.post(
                    f"{api_base}/v1/chat/completions", json=payload, timeout=60
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]

                # Clear "Thinking..." and print response
                print("\r" + " " * 20 + "\r", end="")  # Clear the line
                print(f"🤖 Assistant: {assistant_message}")

                # Add assistant response to history
                messages.append({"role": "assistant", "content": assistant_message})

            except requests.exceptions.ConnectionError:
                print(
                    "\r❌ Error: Cannot connect to server. Is serve_model.py running?"
                )
                sys.exit(1)
            except requests.exceptions.Timeout:
                print(
                    "\r⏱️  Request timed out. The model might be processing a long response."
                )
            except Exception as e:
                print(f"\r❌ Error: {e}")
                # Don't add to message history if there was an error
                messages.pop()  # Remove the user message that caused the error

        except KeyboardInterrupt:
            print("\n\nInterrupted! Goodbye! 👋")
            break
        except EOFError:
            print("\n\nGoodbye! 👋")
            break


def chat_with_model_simple(
    api_base: str = "http://localhost:8000", max_tokens: int = 200
):
    """Simple version without conversation history."""
    print("=" * 60)
    print("Chat with Finetuned Model (Simple Mode - No History)")
    print("=" * 60)
    print(f"API: {api_base}/v1/chat/completions")
    print("Press Ctrl+C to exit")
    print("=" * 60)

    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()

            if not user_input:
                continue

            # Prepare single-turn request
            payload = {
                "messages": [{"role": "user", "content": user_input}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            # Make request
            print("🤔 Thinking...", end="", flush=True)
            try:
                response = requests.post(
                    f"{api_base}/v1/chat/completions", json=payload, timeout=60
                )
                response.raise_for_status()

                # Parse and print response
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                print("\r" + " " * 20 + "\r", end="")  # Clear the line
                print(f"🤖 Assistant: {assistant_message}")

            except requests.exceptions.ConnectionError:
                print(
                    "\r❌ Error: Cannot connect to server. Is serve_model.py running?"
                )
                sys.exit(1)
            except Exception as e:
                print(f"\r❌ Error: {e}")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with served model")
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens per response"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="Disable conversation history"
    )

    args = parser.parse_args()

    if args.no_history:
        chat_with_model_simple(args.api_base, args.max_tokens)
    else:
        chat_with_model(args.api_base, args.max_tokens)
