"""
Integration tests for reasoning content extraction in logging_utils.

These tests make real API calls to verify that:
1. Models with reasoning capability include <think> tags
2. Models without reasoning do NOT include <think> tags
"""

import pytest
import os

from dotenv import load_dotenv

from core.lm import get_dspy_lm
from core.logging_utils import _extract_response_with_reasoning

# Load environment variables
load_dotenv()


# Skip these tests by default (run with `RUN_INTEGRATION_TESTS=1 pytest tests/` to not skip)
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="RUN_INTEGRATION_TESTS is not set"
)
@pytest.mark.parametrize(
    "model_name,has_reasoning",
    [
        ("deepseek/deepseek-reasoner", True),
        ("openai/gpt-4.1-nano", False),
        ("deepinfra/deepseek-ai/DeepSeek-V3.2-Exp", True),
    ],
)
def test_reasoning_extraction(model_name, has_reasoning):
    """Test that reasoning is correctly extracted (or not) based on model capabilities."""

    lm = get_dspy_lm(model_name, temperature=0.7)

    messages = [{"role": "user", "content": "what is 1+1"}]

    # Make the call
    result = lm(messages=messages)

    # Check that we got a response
    assert len(result) == 1
    assert isinstance(result[0], str)

    # Get the history entry
    assert hasattr(lm, "history")
    assert len(lm.history) > 0

    last_entry = lm.history[-1]

    # Extract the response with reasoning
    formatted_response = _extract_response_with_reasoning(last_entry)

    if has_reasoning:
        # Verify it contains <think> tags
        assert (
            "<think>" in formatted_response
        ), f"Expected <think> tag in response for {model_name}, but got: {formatted_response[:200]}"
        assert (
            "</think>" in formatted_response
        ), f"Expected </think> tag in response for {model_name}, but got: {formatted_response[:200]}"

        # Verify the structure: <think>...</think> followed by answer
        think_start = formatted_response.find("<think>")
        think_end = formatted_response.find("</think>")

        assert think_start < think_end, "Invalid <think> tag structure"
        assert think_start >= 0, "<think> tag not found"
        assert think_end > 0, "</think> tag not found"

        # Extract the reasoning content
        reasoning = formatted_response[think_start + len("<think>") : think_end].strip()
        answer = formatted_response[think_end + len("</think>") :].strip()

        # Verify both reasoning and answer are non-empty
        assert len(reasoning) > 0, "Reasoning content should not be empty"
        assert len(answer) > 0, "Answer content should not be empty"

        # The answer should mention "2" (answer to 1+1)
        assert "2" in answer, f"Expected answer to contain '2', but got: {answer}"

        print(f"\n✓ {model_name} correctly provides reasoning in <think> tags")
        print(f"  Reasoning length: {len(reasoning)} chars")
        print(f"  Answer: {answer[:100]}")
    else:
        # Verify it does NOT contain <think> tags
        assert (
            "<think>" not in formatted_response
        ), f"Did not expect <think> tag in response for {model_name}, but got: {formatted_response[:200]}"
        assert (
            "</think>" not in formatted_response
        ), f"Did not expect </think> tag in response for {model_name}, but got: {formatted_response[:200]}"

        # The response should still mention "2" (answer to 1+1)
        assert (
            "2" in formatted_response
        ), f"Expected answer to contain '2', but got: {formatted_response}"

        print(f"\n✓ {model_name} correctly does NOT include <think> tags")
        print(f"  Response: {formatted_response[:100]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
