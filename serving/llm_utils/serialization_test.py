#!/usr/bin/env python3
"""
Tests for serialization utilities
"""

import pytest
from .serialization import (
    TokenSerializer,
    TokenPositionSerializer, 
    TokensSerializer,
    LLMResponseSerializer,
)
from .dataclasses import Token, TokenPosition, Tokens, LLMResponse


class TestTokenSerializer:
    """Test cases for TokenSerializer"""
    
    def test_serialize_token_with_logprob(self):
        """Test serializing a token with logprob"""
        token = Token(token="hello", token_id=123, logprob=-1.5)
        result = TokenSerializer.serialize(token)
        
        expected = {
            "token": "hello",
            "token_id": 123,
            "logprob": -1.5,
        }
        assert result == expected
    
    def test_serialize_token_without_logprob(self):
        """Test serializing a token without logprob (None)"""
        token = Token(token="world", token_id=456, logprob=None)
        result = TokenSerializer.serialize(token)
        
        expected = {
            "token": "world",
            "token_id": 456,
            "logprob": None,
        }
        assert result == expected
    
    def test_deserialize_token_with_logprob(self):
        """Test deserializing token data with logprob"""
        data = {
            "token": "hello",
            "token_id": 123,
            "logprob": -1.5,
        }
        result = TokenSerializer.deserialize(data)
        
        assert result.token == "hello"
        assert result.token_id == 123
        assert result.logprob == -1.5
    
    def test_deserialize_token_without_logprob(self):
        """Test deserializing token data without logprob"""
        data = {
            "token": "world",
            "token_id": 456,
            "logprob": None,
        }
        result = TokenSerializer.deserialize(data)
        
        assert result.token == "world"
        assert result.token_id == 456
        assert result.logprob is None
    
    def test_round_trip_serialization(self):
        """Test that serialize + deserialize returns original token"""
        original = Token(token="test", token_id=789, logprob=-2.3)
        serialized = TokenSerializer.serialize(original)
        deserialized = TokenSerializer.deserialize(serialized)
        
        assert deserialized.token == original.token
        assert deserialized.token_id == original.token_id
        assert deserialized.logprob == original.logprob


class TestTokenPositionSerializer:
    """Test cases for TokenPositionSerializer"""
    
    def test_serialize_token_position(self):
        """Test serializing a token position with top tokens"""
        sampled = Token(token="the", token_id=1, logprob=-0.5)
        top_tokens = [
            Token(token="a", token_id=2, logprob=-1.0),
            Token(token="an", token_id=3, logprob=-1.5),
        ]
        token_pos = TokenPosition(sampled_token=sampled, top_tokens=top_tokens)
        
        result = TokenPositionSerializer.serialize(token_pos)
        
        assert result["sampled_token"]["token"] == "the"
        assert len(result["top_tokens"]) == 2
        assert result["top_tokens"][0]["token"] == "a"
        assert result["top_tokens"][1]["token"] == "an"
    
    def test_serialize_token_position_empty_top_tokens(self):
        """Test serializing a token position with no top tokens"""
        sampled = Token(token="hello", token_id=1, logprob=-0.5)
        token_pos = TokenPosition(sampled_token=sampled, top_tokens=[])
        
        result = TokenPositionSerializer.serialize(token_pos)
        
        assert result["sampled_token"]["token"] == "hello"
        assert result["top_tokens"] == []
    
    def test_round_trip_token_position(self):
        """Test round-trip serialization for TokenPosition"""
        sampled = Token(token="test", token_id=1, logprob=-0.5)
        top_tokens = [Token(token="best", token_id=2, logprob=-1.0)]
        original = TokenPosition(sampled_token=sampled, top_tokens=top_tokens)
        
        serialized = TokenPositionSerializer.serialize(original)
        deserialized = TokenPositionSerializer.deserialize(serialized)
        
        assert deserialized.sampled_token.token == "test"
        assert len(deserialized.top_tokens) == 1
        assert deserialized.top_tokens[0].token == "best"


class TestTokensSerializer:
    """Test cases for TokensSerializer"""
    
    def test_serialize_tokens(self):
        """Test serializing a Tokens object"""
        pos1 = TokenPosition(
            sampled_token=Token("hello", 1, -0.5),
            top_tokens=[Token("hi", 2, -1.0)]
        )
        pos2 = TokenPosition(
            sampled_token=Token("world", 3, -0.3),
            top_tokens=[]
        )
        tokens = Tokens(tokens=[pos1, pos2], text="hello world")
        
        result = TokensSerializer.serialize(tokens)
        
        assert result["text"] == "hello world"
        assert len(result["tokens"]) == 2
        assert result["tokens"][0]["sampled_token"]["token"] == "hello"
        assert result["tokens"][1]["sampled_token"]["token"] == "world"
    
    def test_round_trip_tokens(self):
        """Test round-trip serialization for Tokens"""
        pos = TokenPosition(
            sampled_token=Token("test", 1, -0.5),
            top_tokens=[Token("rest", 2, -1.0)]
        )
        original = Tokens(tokens=[pos], text="test")
        
        serialized = TokensSerializer.serialize(original)
        deserialized = TokensSerializer.deserialize(serialized)
        
        assert deserialized.text == "test"
        assert len(deserialized.tokens) == 1
        assert deserialized.tokens[0].sampled_token.token == "test"


class TestLLMResponseSerializer:
    """Test cases for LLMResponseSerializer"""
    
    def test_serialize_llm_response(self):
        """Test serializing a complete LLMResponse"""
        pos = TokenPosition(
            sampled_token=Token("hello", 1, -0.5),
            top_tokens=[Token("hi", 2, -1.0)]
        )
        tokens = Tokens(tokens=[pos], text="hello")
        response = LLMResponse(
            system_prompt="You are helpful",
            user_prompt="Say hello",
            response=tokens
        )
        
        result = LLMResponseSerializer.serialize(response)
        
        assert result["system_prompt"] == "You are helpful"
        assert result["user_prompt"] == "Say hello"
        assert result["response"]["text"] == "hello"
    
    def test_round_trip_llm_response(self):
        """Test round-trip serialization for LLMResponse"""
        pos = TokenPosition(
            sampled_token=Token("test", 1, -0.5),
            top_tokens=[]
        )
        tokens = Tokens(tokens=[pos], text="test")
        original = LLMResponse(
            system_prompt="Test system",
            user_prompt="Test user",
            response=tokens
        )
        
        serialized = LLMResponseSerializer.serialize(original)
        deserialized = LLMResponseSerializer.deserialize(serialized)
        
        assert deserialized.system_prompt == "Test system"
        assert deserialized.user_prompt == "Test user"
        assert deserialized.response.text == "test"
        assert len(deserialized.response.tokens) == 1
    
    def test_serialize_with_empty_prompts(self):
        """Test serializing with empty system prompt"""
        pos = TokenPosition(
            sampled_token=Token("ok", 1, -0.5),
            top_tokens=[]
        )
        tokens = Tokens(tokens=[pos], text="ok")
        response = LLMResponse(
            system_prompt="",
            user_prompt="test",
            response=tokens
        )
        
        result = LLMResponseSerializer.serialize(response)
        
        assert result["system_prompt"] == ""
        assert result["user_prompt"] == "test"