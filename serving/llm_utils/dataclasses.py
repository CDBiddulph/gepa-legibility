#!/usr/bin/env python3
"""
Core dataclasses for LLM operations

Simplified to only support text generation without token-level details.
"""

from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Represents a response from an LLM.

    Args:
        system_prompt: The system prompt used.
        user_prompt: The user prompt used.
        response_text: The generated text response.
    """

    system_prompt: str
    user_prompt: str
    response_text: str

    def __hash__(self):
        return abs(hash((self.system_prompt, self.user_prompt, self.response_text)))
