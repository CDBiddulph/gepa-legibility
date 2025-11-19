#!/usr/bin/env python3
"""
Simple test script to verify local model setup works end-to-end.

Usage:
    python test_local_model.py
"""

import sys
sys.path.insert(0, '/nas/ucb/biddulph/gepa-legibility')

# Load .env file for API keys (required by litellm even for local models)
from dotenv import load_dotenv
load_dotenv('/nas/ucb/biddulph/gepa-legibility/.env')

from lm import get_dspy_lm

def test_local_model():
    """Test local model via filesystem queue."""
    print("Testing local model via filesystem queue...")
    print("Make sure:")
    print("1. vLLM worker is running (./start_vllm_worker.sh)")
    print("2. API server is running (python api_server.py)")
    print()

    # Create LM with local/ prefix
    model_name = "local/allenai/OLMo-2-1124-13B-SFT"
    print(f"Creating LM with model: {model_name}")

    lm = get_dspy_lm(model_name, temperature=0.7, max_tokens=100)

    # Simple test
    print("\nTest 1: Simple prompt")
    print("Prompt: 'What is 2+2?'")
    response = lm("What is 2+2?")
    print(f"Response: {response}")

    # Test with messages
    print("\nTest 2: Messages format")
    messages = [
        {"role": "system", "content": "You are a helpful math tutor"},
        {"role": "user", "content": "Explain what prime numbers are"}
    ]
    print(f"Messages: {messages}")
    response = lm(messages=messages)
    print(f"Response: {response}")

    print("\n✅ Tests completed successfully!")

if __name__ == "__main__":
    try:
        test_local_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
