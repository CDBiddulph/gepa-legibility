#!/usr/bin/env python3
"""
Core dataclasses for LLM operations

This module contains the fundamental data structures used throughout the LLM system,
including tokens, responses, and related utilities.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Token:
    """Represents a token and its log probability.

    Args:
        token: The token as a string.
        token_id: The token as an integer.
        logprob: The log probability of the token. Should be None if this token is part of a prompt.
    """

    token: str
    token_id: int
    logprob: float | None

    def __hash__(self):
        return hash((self.token, self.token_id, self.logprob))


@dataclass
class TokenPosition:
    """Represents the sampled and top tokens at a given position in a response.

    Args:
        sampled_token: The sampled token.
        top_tokens: A list of the top tokens (may be empty).
    """

    sampled_token: Token
    top_tokens: List[Token]

    def __hash__(self):
        return hash((self.sampled_token, *self.top_tokens))


@dataclass
class Tokens:
    """Represents a series of tokens and their log probabilities.

    Args:
        tokens: A list of TokenPosition objects.
        text: The text that the tokens represent.
    """

    tokens: List[TokenPosition]
    text: str

    def __hash__(self):
        return hash((self.text, *self.tokens))


@dataclass
class LLMResponse:
    """Represents a response from an LLM.

    Args:
        system_prompt: The system prompt used.
        user_prompt: The user prompt used.
        response: The response tokens.
    """

    system_prompt: str
    user_prompt: str
    response: Tokens

    def __hash__(self):
        return abs(hash((self.system_prompt, self.user_prompt, self.response)))
