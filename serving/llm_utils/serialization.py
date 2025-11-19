#!/usr/bin/env python3
"""
Serialization utilities for LLM interfaces

Provides shared serialization/deserialization functionality for Token, TokenPosition,
Tokens, and LLMResponse objects to eliminate code duplication across modules.
"""

from typing import Dict, Any, List
from .dataclasses import Token, TokenPosition, Tokens, LLMResponse


class TokenSerializer:
    """Handles serialization and deserialization of Token objects"""
    
    @staticmethod
    def serialize(token: Token) -> Dict[str, Any]:
        """Serialize a Token to JSON-compatible dict"""
        return {
            "token": token.token,
            "token_id": token.token_id,
            "logprob": token.logprob,
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Token:
        """Deserialize dict data back to Token"""
        return Token(
            token=data["token"],
            token_id=data["token_id"],
            logprob=data["logprob"],
        )


class TokenPositionSerializer:
    """Handles serialization and deserialization of TokenPosition objects"""
    
    @staticmethod
    def serialize(token_position: TokenPosition) -> Dict[str, Any]:
        """Serialize a TokenPosition to JSON-compatible dict"""
        return {
            "sampled_token": TokenSerializer.serialize(token_position.sampled_token),
            "top_tokens": [
                TokenSerializer.serialize(token) for token in token_position.top_tokens
            ],
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> TokenPosition:
        """Deserialize dict data back to TokenPosition"""
        sampled_token = TokenSerializer.deserialize(data["sampled_token"])
        top_tokens = [
            TokenSerializer.deserialize(token_data) for token_data in data["top_tokens"]
        ]
        return TokenPosition(sampled_token=sampled_token, top_tokens=top_tokens)


class TokensSerializer:
    """Handles serialization and deserialization of Tokens objects"""
    
    @staticmethod
    def serialize(tokens: Tokens) -> Dict[str, Any]:
        """Serialize a Tokens object to JSON-compatible dict"""
        return {
            "tokens": [
                TokenPositionSerializer.serialize(pos) for pos in tokens.tokens
            ],
            "text": tokens.text,
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Tokens:
        """Deserialize dict data back to Tokens"""
        token_positions = [
            TokenPositionSerializer.deserialize(pos_data) for pos_data in data["tokens"]
        ]
        return Tokens(tokens=token_positions, text=data["text"])


class LLMResponseSerializer:
    """Handles serialization and deserialization of LLMResponse objects"""
    
    @staticmethod
    def serialize(response: LLMResponse) -> Dict[str, Any]:
        """Serialize an LLMResponse to JSON-compatible dict"""
        return {
            "system_prompt": response.system_prompt,
            "user_prompt": response.user_prompt,
            "response": TokensSerializer.serialize(response.response),
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> LLMResponse:
        """Deserialize dict data back to LLMResponse"""
        response_tokens = TokensSerializer.deserialize(data["response"])
        return LLMResponse(
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"],
            response=response_tokens,
        )