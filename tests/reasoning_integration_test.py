"""
Integration tests for reasoning content extraction in logging_utils.

These tests make real API calls to verify that:
1. deepseek/deepseek-reasoner responses include <think> tags
2. gpt-4.1-nano responses do NOT include <think> tags
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from lm import get_dspy_lm
from logging_utils import _extract_response_with_reasoning

# Load environment variables
load_dotenv()


# Skip these tests for now
@pytest.mark.skip(reason="Skipping these tests to avoid API calls")
class TestReasoningIntegration:
    """Integration tests for reasoning models."""

    def test_deepseek_reasoner_has_think_tags(self):
        """Test that deepseek/deepseek-reasoner provides reasoning in <think> tags."""

        model_name = "deepseek/deepseek-reasoner"
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

    def test_gpt_4_1_nano_no_think_tags(self):
        """Test that gpt-4.1-nano does NOT have <think> tags (no reasoning_content)."""

        model_name = "openai/gpt-4.1-nano"
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

        # Extract the response
        formatted_response = _extract_response_with_reasoning(last_entry)

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
