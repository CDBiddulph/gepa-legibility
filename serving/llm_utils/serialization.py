#!/usr/bin/env python3
"""
Serialization utilities for LLM interfaces

Simplified to only handle text-based LLMResponse objects.
"""

from typing import Dict, Any
from .dataclasses import LLMResponse


class LLMResponseSerializer:
    """Handles serialization and deserialization of LLMResponse objects"""

    @staticmethod
    def serialize(response: LLMResponse) -> Dict[str, Any]:
        """Serialize an LLMResponse to JSON-compatible dict"""
        return {
            "system_prompt": response.system_prompt,
            "user_prompt": response.user_prompt,
            "response_text": response.response_text,
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> LLMResponse:
        """Deserialize dict data back to LLMResponse"""
        return LLMResponse(
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"],
            response_text=data["response_text"],
        )
