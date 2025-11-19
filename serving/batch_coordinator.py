#!/usr/bin/env python3
"""
Async Batch Coordinator

This module provides async batching functionality that collects LLM requests
across multiple LLM objects and batches them together when they use the same
underlying model weights.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from .base import LLMResponse
import time


@dataclass
class BatchRequest:
    """Represents a single request in a batch"""

    request_type: str  # "generate_response" | "get_logprobs"
    params: Dict[str, Any]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


@dataclass
class BatchConfig:
    """Configuration for batching behavior"""

    enabled: bool = True
    default_delay_ms: int = 50
    max_batch_size: int = 32
    per_model_delays: Dict[str, int] = field(default_factory=dict)


class AsyncBatchCoordinator:
    """Coordinates async batching of LLM requests by model identifier"""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.pending_batches: Dict[str, List[BatchRequest]] = {}
        self.batch_timers: Dict[str, asyncio.Task] = {}
        self.batch_executors: Dict[
            str, Callable[[str, List[BatchRequest]], Awaitable[List[LLMResponse]]]
        ] = {}
        self._lock = asyncio.Lock()

        logging.info(
            f"Initialized AsyncBatchCoordinator with default delay {self.config.default_delay_ms}ms"
        )

    def register_batch_executor(
        self,
        model_id: str,
        executor: Callable[[str, List[BatchRequest]], Awaitable[List[LLMResponse]]],
    ):
        """Register a batch executor function for a specific model ID"""
        if model_id in self.batch_executors:
            logging.debug(
                f"Batch executor for model {model_id} already registered, overriding"
            )
        self.batch_executors[model_id] = executor
        logging.debug(f"Registered batch executor for model: {model_id}")

    async def submit_request(
        self, model_id: str, request_type: str, params: Dict[str, Any]
    ) -> LLMResponse:
        """Submit a request for batching and return a future for the response"""
        if not self.config.enabled:
            # If batching is disabled, execute immediately using sync fallback
            if model_id not in self.batch_executors:
                raise ValueError(f"No batch executor registered for model: {model_id}")

            # Create a single-item batch and execute
            future = asyncio.Future()
            request = BatchRequest(request_type, params, future)
            responses = await self.batch_executors[model_id](model_id, [request])
            return responses[0]

        # Create future for this request
        future = asyncio.Future()
        request = BatchRequest(request_type, params, future)

        async with self._lock:
            # Add to pending batch
            if model_id not in self.pending_batches:
                self.pending_batches[model_id] = []

            self.pending_batches[model_id].append(request)

            # Check if we need to start a new timer or execute immediately
            if len(self.pending_batches[model_id]) >= self.config.max_batch_size:
                # Execute immediately when batch is full
                await self._execute_batch_now(model_id)
            elif model_id not in self.batch_timers:
                # Start a new timer for this model
                delay_ms = self.config.per_model_delays.get(
                    model_id, self.config.default_delay_ms
                )
                delay_seconds = delay_ms / 1000.0

                self.batch_timers[model_id] = asyncio.create_task(
                    self._batch_timer(model_id, delay_seconds)
                )

        # Wait for response
        return await future

    async def _batch_timer(self, model_id: str, delay_seconds: float):
        """Timer that executes a batch after the specified delay"""
        try:
            await asyncio.sleep(delay_seconds)
            async with self._lock:
                await self._execute_batch_now(model_id)
        except asyncio.CancelledError:
            # Timer was cancelled, which is expected when batch executes early
            pass

    async def _execute_batch_now(self, model_id: str):
        """Execute the pending batch for a model immediately (must be called with lock held)"""
        if model_id not in self.pending_batches or not self.pending_batches[model_id]:
            return

        # Get and clear the pending requests
        requests = self.pending_batches[model_id]
        self.pending_batches[model_id] = []

        # Cancel any pending timer
        if model_id in self.batch_timers:
            self.batch_timers[model_id].cancel()
            del self.batch_timers[model_id]

        # Execute the batch asynchronously (outside the lock)
        asyncio.create_task(self._execute_batch_async(model_id, requests))

    async def _execute_batch_async(self, model_id: str, requests: List[BatchRequest]):
        """Execute a batch of requests asynchronously"""
        try:
            if model_id not in self.batch_executors:
                # No batch executor - resolve each request with an error
                error = ValueError(
                    f"No batch executor registered for model: {model_id}"
                )
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(error)
                return

            logging.info(
                f"Executing batch of {len(requests)} requests for model: {model_id}"
            )

            # Execute the batch
            start_time = time.time()
            responses = await self.batch_executors[model_id](model_id, requests)
            end_time = time.time()

            logging.debug(
                f"Batch execution took {(end_time - start_time)*1000:.1f}ms for {len(requests)} requests"
            )

            # Verify we got the right number of responses
            if len(responses) != len(requests):
                raise ValueError(
                    f"Batch executor returned {len(responses)} responses for {len(requests)} requests"
                )

            # Resolve all futures with their responses
            for request, response in zip(requests, responses):
                if not request.future.done():
                    request.future.set_result(response)

        except Exception as e:
            logging.error(f"Error executing batch for model {model_id}: {e}")
            # Resolve all futures with the error
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)


# Global coordinator instance
_global_coordinator: Optional[AsyncBatchCoordinator] = None


def get_coordinator() -> AsyncBatchCoordinator:
    """Get the global batch coordinator instance"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = AsyncBatchCoordinator()
    return _global_coordinator


def configure_coordinator(config: BatchConfig):
    """Configure the global batch coordinator"""
    global _global_coordinator

    # Preserve existing batch executors if coordinator already exists
    existing_executors = {}
    if _global_coordinator is not None:
        existing_executors = _global_coordinator.batch_executors.copy()

    _global_coordinator = AsyncBatchCoordinator(config)

    # Re-register existing batch executors
    for model_id, executor in existing_executors.items():
        _global_coordinator.register_batch_executor(model_id, executor)
