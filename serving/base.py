#!/usr/bin/env python3
"""
Base LLM Interface Definitions

This module provides abstract interfaces and data structures for LLM providers.
"""

import enum
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from llm_utils.dataclasses import Token, TokenPosition, Tokens, LLMResponse
from .batch_coordinator import get_coordinator


class LLMDefaults:
    """Default values for LLM settings"""

    DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class LLMConfig:
    """Specifies an LLM to be initialized"""

    model: str
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    # For cached LLM configurations
    inner_llm: Optional[Union[Dict[str, Any], "LLMConfig"]] = None
    use_cache_for_reads: Optional[bool] = None


class LLMMethod(enum.Enum):
    """Enum for LLM methods"""

    GENERATE_RESPONSE = "generate_response"
    GET_LOGPROBS = "get_logprobs"


class LLM(ABC):
    """Abstract base class for LLM interfaces"""

    def __init__(self, system_prompt: Optional[str] = None, name: Optional[str] = None):
        self.system_prompt = system_prompt
        self.name = name or "LLM"
        self.call_count = 0

    def generate_response(
        self, prompt: str, partial_response: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to generate a response for.
            partial_response: If provided, continue from this partial response instead of generating from scratch.

        Returns:
            A LLMResponse object containing the response text and tokens (without logprobs).
        """
        method_name = LLMMethod.GENERATE_RESPONSE
        self._log_input(method_name, prompt=prompt, partial_response=partial_response)
        response = self._generate_response_impl(prompt, partial_response)
        self._validate_response(method_name, response)
        self._log_response(method_name, response)
        self.call_count += 1
        return response

    def get_logprobs(
        self, old_response: LLMResponse, top_logprobs: int = 0
    ) -> LLMResponse:
        """Create a new LLMResponse object, with log probabilities in the response taken from this LLM"""
        method_name = LLMMethod.GET_LOGPROBS
        self._log_input(method_name, response=old_response, top_logprobs=top_logprobs)
        new_response = self._get_logprobs_impl(old_response, top_logprobs)
        self._validate_response(method_name, new_response)
        self._validate_responses_match(old_response, new_response)
        self._log_response(method_name, new_response)
        self.call_count += 1
        return new_response

    @abstractmethod
    def _generate_response_impl(
        self, prompt: str, partial_response: Optional[str] = None
    ) -> LLMResponse:
        """Implementation of generate_response - to be overridden by subclasses"""
        pass

    @abstractmethod
    def _get_logprobs_impl(
        self, old_response: LLMResponse, top_logprobs: int = 0
    ) -> LLMResponse:
        """Implementation of get_logprobs - to be overridden by subclasses"""
        pass

    @abstractmethod
    def get_model_info(self) -> str:
        """Get the model information"""
        pass

    def get_batch_model_id(self) -> str:
        """Get the model identifier for batching purposes.

        By default, uses get_model_info(), but subclasses can override
        to provide a different identifier for batching.
        """
        return self.get_model_info()

    async def generate_response_async(
        self, prompt: str, partial_response: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response from the LLM asynchronously with automatic batching.

        Args:
            prompt: The prompt to generate a response for.
            partial_response: If provided, continue from this partial response instead of generating from scratch.

        Returns:
            A LLMResponse object containing the response text and tokens (without logprobs).
        """
        coordinator = get_coordinator()
        return await coordinator.submit_request(
            self.get_batch_model_id(),
            "generate_response",
            {
                "prompt": prompt,
                "system_prompt": self.system_prompt,
                "partial_response": partial_response,
            },
        )

    async def get_logprobs_async(
        self, old_response: LLMResponse, top_logprobs: int = 0
    ) -> LLMResponse:
        """Create a new LLMResponse object asynchronously with log probabilities taken from this LLM.

        Args:
            old_response: The response to get logprobs for.
            top_logprobs: Number of top log probabilities to include.

        Returns:
            A new LLMResponse object with log probabilities.
        """
        coordinator = get_coordinator()
        return await coordinator.submit_request(
            self.get_batch_model_id(),
            "get_logprobs",
            {
                "old_response": old_response,
                "top_logprobs": top_logprobs,
                "system_prompt": self.system_prompt,
            },
        )

    def _validate_logprobs_do_not_exist(self, tokens: Tokens, tokens_name: str):
        """Validate that there are no logprobs in the tokens"""
        for i, token_pos in enumerate(tokens.tokens):
            if token_pos.sampled_token.logprob is not None:
                raise ValueError(
                    f"Token at position {i} should not have logprob, but got {token_pos.sampled_token.logprob}"
                )
            if token_pos.top_tokens:
                raise ValueError(
                    f"Token at position {i} in {tokens_name} should not have top_tokens, but got {len(token_pos.top_tokens)} tokens"
                )

    def _validate_logprobs_exist(self, tokens: Tokens, tokens_name: str):
        """Validate that there are logprobs in the tokens"""
        for i, token_pos in enumerate(tokens.tokens):
            if token_pos.sampled_token.logprob is None:
                raise ValueError(
                    f"Token at position {i} in {tokens_name} should have logprob, but got None"
                )
            # top_tokens may be empty, but if it's not, there should be logprobs
            for j, top_token in enumerate(token_pos.top_tokens):
                if top_token.logprob is None:
                    raise ValueError(
                        f"Token at position {i} in {tokens_name}, top token {j} should have logprob, but got None"
                    )

    def _validate_response(self, method_name: LLMMethod, response: LLMResponse):
        """Validate the response"""
        # Check that there are/are not logprobs in the response
        if method_name == LLMMethod.GENERATE_RESPONSE:
            self._validate_logprobs_do_not_exist(response.response, "response")
        elif method_name == LLMMethod.GET_LOGPROBS:
            self._validate_logprobs_exist(response.response, "response")
        else:
            raise ValueError(f"Invalid method name: {method_name}")

    def _validate_responses_match(
        self, old_response: LLMResponse, new_response: LLMResponse
    ):
        """Validate that the responses match"""
        # The user prompt should match (but system prompt can be different for get_logprobs)
        if old_response.user_prompt != new_response.user_prompt:
            raise ValueError(
                f"User prompt should match: old='{old_response.user_prompt}' vs new='{new_response.user_prompt}'"
            )

        # The response may differ in its logprobs, but the sampled tokens should be the same
        if old_response.response.text != new_response.response.text:
            raise ValueError(
                f"Response text should be identical: old={json.dumps(old_response.response.text)} vs new={json.dumps(new_response.response.text)}"
            )
        old_tokens = [tp.sampled_token.token for tp in old_response.response.tokens]
        new_tokens = [tp.sampled_token.token for tp in new_response.response.tokens]
        old_token_ids = [
            tp.sampled_token.token_id for tp in old_response.response.tokens
        ]
        new_token_ids = [
            tp.sampled_token.token_id for tp in new_response.response.tokens
        ]
        if len(old_tokens) != len(new_tokens):
            raise ValueError(
                f"Response should have same number of tokens: old={old_tokens} ({len(old_tokens)}) vs new={new_tokens} ({len(new_tokens)})"
            )
        for i, (old_token, new_token, old_token_id, new_token_id) in enumerate(
            zip(old_tokens, new_tokens, old_token_ids, new_token_ids, strict=True)
        ):
            if old_token != new_token or old_token_id != new_token_id:
                raise ValueError(
                    f"Response token at position {i} should match: old={json.dumps(old_token)} (ID = {old_token_id}, from {old_tokens}) vs new={json.dumps(new_token)} (ID = {new_token_id}, from {new_tokens})"
                )

    def _log_input(
        self,
        method_name: LLMMethod,
        response: Optional[LLMResponse] = None,
        prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        partial_response: Optional[str] = None,
    ):
        """Log LLM input in a simple, readable format"""
        logger = logging.getLogger(f"{self.__class__.__name__}")

        if not logger.isEnabledFor(logging.INFO):
            return

        # Only show the response and prompt text for generate_response
        # For get_logprobs, the hash is enough to identify the response from the previous generate_response call
        logger.info(
            f'=== Starting "{self.name}" call #{self.call_count + 1} ({method_name.value}) ==='
        )
        if prompt:
            logger.info(f"Prompt text: {repr(prompt)}")
        if partial_response:
            logger.info(f"Partial response: {repr(partial_response)}")
        if top_logprobs is not None:
            logger.info(f"Top logprobs: {top_logprobs}")
        if response:
            if method_name == LLMMethod.GENERATE_RESPONSE:
                logger.info(f"Response text: {response.response.text}")
            logger.info(f"Input hash:  {hash(response)}")

    def _log_response(self, method_name: LLMMethod, response: LLMResponse):
        """Log LLM response in a simple, readable format"""
        logger = logging.getLogger(f"{self.__class__.__name__}")

        if not logger.isEnabledFor(logging.INFO):
            return

        logger.info(
            f'=== Finished "{self.name}" call #{self.call_count + 1} ({method_name.value}) ==='
        )
        # Only show the response text for generate_response
        # For get_logprobs, the hash is enough to identify the response from the previous generate_response call
        if method_name == LLMMethod.GENERATE_RESPONSE:
            logger.info(f"Response text: {response.response.text}")
            logger.info(f"Number of tokens: {len(response.response.tokens)}")
        logger.info(f"Output hash: {hash(response)}")
        logger.info("=" * 50)
