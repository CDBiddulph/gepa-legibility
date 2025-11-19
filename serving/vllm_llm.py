#!/usr/bin/env python3
"""
vLLM Interface Implementation

This module provides a vLLM interface that communicates with a remote vLLM worker
process via a file-based job queue system.
"""

import asyncio
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .base import LLM, LLMDefaults
from .batch_coordinator import BatchRequest
from llm_utils.dataclasses import LLMResponse, Token, TokenPosition, Tokens
from llm_utils.serialization import LLMResponseSerializer


class VllmLlm(LLM):
    """Interface for vLLM models via file-based communication"""

    def __init__(
        self,
        model_id: str = LLMDefaults.DEFAULT_VLLM_MODEL,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 128,
        queue_dir: str = "/tmp/vllm_queue",
        timeout: float = 300.0,
        name: Optional[str] = None,
    ):
        super().__init__(system_prompt, name)
        self.queue_dir = Path(queue_dir)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Create queue directories if they don't exist
        self.requests_dir = self.queue_dir / "requests"
        self.processing_dir = self.queue_dir / "processing"
        self.responses_dir = self.queue_dir / "responses"

        for dir_path in [self.requests_dir, self.processing_dir, self.responses_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Initialized VllmLlm with queue_dir: {queue_dir}")

        # Register this instance as a batch executor
        from .batch_coordinator import get_coordinator

        coordinator = get_coordinator()
        coordinator.register_batch_executor(
            self.get_batch_model_id(), self._execute_batch
        )

    def _validate_worker_availability(self) -> None:
        """Validate that there's an active worker for this model"""
        heartbeat_dir = self.queue_dir / "heartbeat"

        if not heartbeat_dir.exists():
            raise RuntimeError(
                f"No vLLM workers found: heartbeat directory {heartbeat_dir} does not exist"
            )

        # Check for active workers running our model
        current_time = time.time()
        worker_timeout = (
            300.0  # 5 minutes - workers should update heartbeat every minute
        )

        active_workers = []
        matching_workers = []

        try:
            for heartbeat_file in heartbeat_dir.glob("*.json"):
                try:
                    with open(heartbeat_file, "r") as f:
                        heartbeat_data = json.load(f)

                    # Check if worker is recent enough to be considered active
                    if current_time - heartbeat_data["timestamp"] < worker_timeout:
                        active_workers.append(heartbeat_data["model_id"])

                        # Check if this worker matches our model
                        if heartbeat_data["model_id"] == self.model_id:
                            matching_workers.append(heartbeat_data["worker_id"])

                except (json.JSONDecodeError, KeyError, OSError) as e:
                    # Skip malformed or unreadable heartbeat files
                    logging.debug(
                        f"Skipping invalid heartbeat file {heartbeat_file}: {e}"
                    )
                    continue

        except OSError as e:
            # Handle directory access issues
            raise RuntimeError(
                f"Cannot access worker heartbeat directory {heartbeat_dir}: {e}"
            )

        if not matching_workers:
            if not active_workers:
                raise RuntimeError(
                    f"No active vLLM workers found. "
                    f"Start a worker for model '{self.model_id}' using the start_vllm_worker.sh script."
                )
            else:
                active_models_str = ", ".join(
                    f"'{model}'" for model in sorted(set(active_workers))
                )
                raise RuntimeError(
                    f"No vLLM worker found for model '{self.model_id}'. "
                    f"Found active workers for: {active_models_str}. "
                    f"Start a worker for '{self.model_id}' or use one of the available models."
                )

        logging.debug(
            f"Found {len(matching_workers)} active worker(s) for model '{self.model_id}'"
        )

    def _submit_job(self, method: str, params: Dict[str, Any]) -> str:
        """Submit a job to the worker queue and return job ID"""
        # Validate worker availability before submitting
        self._validate_worker_availability()

        job_id = f"{method}_{uuid.uuid4().hex}"

        request_data = {
            "job_id": job_id,
            "method": method,
            "model_id": self.model_id,  # Add model_id for worker validation
            "params": params,
            "timestamp": time.time(),
        }

        request_file = self.requests_dir / f"{job_id}.json"

        with open(request_file, "w") as f:
            json.dump(request_data, f)

        logging.debug(f"Submitted job {job_id}: {method}")
        return job_id

    def _refresh_directory_cache(self):
        """Force NFS directory cache refresh"""
        try:
            list(self.responses_dir.iterdir())
        except OSError:
            # Ignore stale file handle errors during directory refresh
            pass

    def _process_response_file(
        self, job_id: str, response_file: Path
    ) -> Dict[str, Any]:
        """Process and validate response file, returning response data"""
        try:
            with open(response_file, "r") as f:
                response_data = json.load(f)

            # Clean up response file
            try:
                response_file.unlink()
            except OSError:
                # Ignore stale file handle errors during cleanup
                pass

            if response_data["status"] == "error":
                raise RuntimeError(f"Worker error: {response_data['error']}")

            logging.debug(f"Received response for job {job_id}")
            return response_data["response"]

        except OSError as e:
            # Handle stale file handle or other filesystem errors
            logging.warning(
                f"Filesystem error reading response file for job {job_id}: {e}"
            )
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid response file for job {job_id}: {e}")
            try:
                response_file.unlink()  # Clean up invalid file
            except OSError:
                # Ignore stale file handle errors during cleanup
                pass
            return None

    def _wait_for_response(self, job_id: str) -> Dict[str, Any]:
        """Wait for and retrieve job response"""
        response_file = self.responses_dir / f"{job_id}.json"
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            self._refresh_directory_cache()

            if response_file.exists():
                result = self._process_response_file(job_id, response_file)
                if result is not None:
                    return result

            time.sleep(0.2)  # Poll every 200ms

        raise TimeoutError(f"Job {job_id} timed out after {self.timeout} seconds")

    def _generate_response_impl(
        self, prompt: str, partial_response: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response from the LLM via worker"""
        params = {
            "model_id": self.model_id,  # Add model_id for worker validation
            "user_prompt": prompt,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "partial_response": partial_response,
        }

        job_id = self._submit_job("generate_response", params)
        response_data = self._wait_for_response(job_id)
        return LLMResponseSerializer.deserialize(response_data)

    def _get_logprobs_impl(
        self, old_response: LLMResponse, top_logprobs: int = 0
    ) -> LLMResponse:
        """Get the log probabilities of the tokens in a fixed response via worker"""
        # Extract the token IDs from the original response to ensure consistency
        response_token_ids = [
            tp.sampled_token.token_id for tp in old_response.response.tokens
        ]
        params = {
            "model_id": self.model_id,  # Add model_id for worker validation
            "response_token_ids": response_token_ids,
            "user_prompt": old_response.user_prompt,
            "system_prompt": self.system_prompt,
            "top_logprobs": top_logprobs,
        }

        job_id = self._submit_job("get_logprobs", params)
        response_data = self._wait_for_response(job_id)
        return LLMResponseSerializer.deserialize(response_data)

    def get_model_info(self) -> str:
        """Get the model information"""
        return f"vllm/{self.model_id}"

    def check_worker_health(self) -> bool:
        """Check if any workers are active for this model"""
        heartbeat_dir = self.queue_dir / "heartbeat"
        if not heartbeat_dir.exists():
            return False

        current_time = time.time()
        for heartbeat_file in heartbeat_dir.glob("*.json"):
            try:
                with open(heartbeat_file, "r") as f:
                    heartbeat_data = json.load(f)

                # Check if heartbeat is recent (within last 60 seconds) and matches our model
                if (
                    heartbeat_data.get("model_id") == self.model_id
                    and current_time - heartbeat_data.get("timestamp", 0) < 60
                ):
                    return True

            except (json.JSONDecodeError, IOError):
                continue

        return False

    def cleanup_old_files(self, max_age_hours: float = 24.0):
        """Clean up old request/response files"""
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()

        for directory in [self.requests_dir, self.responses_dir]:
            for file_path in directory.glob("*.json"):
                try:
                    if current_time - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()
                        logging.debug(f"Cleaned up old file: {file_path}")
                except (OSError, IOError):
                    continue

    def get_batch_model_id(self) -> str:
        """Use model_id for batching instead of full get_model_info()

        This allows multiple VllmLlm instances with the same model but different
        system prompts to share the same batch executor for efficiency.
        """
        return self.model_id

    async def _execute_batch(self, model_id: str, requests) -> List[LLMResponse]:
        """Execute a batch of requests for VllmLlm.

        This provides the main batching benefit by submitting multiple jobs
        concurrently and waiting for all responses.
        """
        # Log the batch composition
        request_types = [req.request_type for req in requests]
        request_summary = {}
        for req_type in request_types:
            request_summary[req_type] = request_summary.get(req_type, 0) + 1

        summary_str = ", ".join(
            [f"{count} {req_type}" for req_type, count in request_summary.items()]
        )
        logging.info(f"VllmLlm executing batch: {summary_str} (model: {model_id})")
        # Submit all jobs concurrently
        job_tasks = []
        for i, request in enumerate(requests):
            if request.request_type == "generate_response":
                params = {
                    "model_id": self.model_id,
                    "user_prompt": request.params["prompt"],
                    "system_prompt": request.params["system_prompt"],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                job_id = self._submit_job("generate_response", params)
                logging.debug(
                    f"Submitted generate_response job {job_id} (batch item {i+1}/{len(requests)})"
                )
                job_tasks.append(
                    asyncio.create_task(self._wait_for_response_async(job_id))
                )
            elif request.request_type == "get_logprobs":
                # Extract token IDs from the old response
                old_response = request.params["old_response"]
                response_token_ids = [
                    tp.sampled_token.token_id for tp in old_response.response.tokens
                ]
                params = {
                    "model_id": self.model_id,  # Add model_id for worker validation
                    "response_token_ids": response_token_ids,
                    "user_prompt": old_response.user_prompt,
                    "system_prompt": request.params["system_prompt"],
                    "top_logprobs": request.params["top_logprobs"],
                }
                job_id = self._submit_job("get_logprobs", params)
                logging.debug(
                    f"Submitted get_logprobs job {job_id} (batch item {i+1}/{len(requests)})"
                )
                job_tasks.append(
                    asyncio.create_task(self._wait_for_response_async(job_id))
                )
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")

        # Wait for all responses concurrently
        logging.info(f"Waiting for {len(job_tasks)} vLLM jobs to complete...")
        start_time = time.time()
        response_data_list = await asyncio.gather(*job_tasks)
        end_time = time.time()

        logging.info(
            f"All {len(job_tasks)} vLLM jobs completed in {(end_time - start_time)*1000:.1f}ms"
        )
        # Deserialize all responses
        responses = []
        for response_data in response_data_list:
            responses.append(LLMResponseSerializer.deserialize(response_data))

        return responses

    async def _wait_for_response_async(self, job_id: str) -> Dict[str, Any]:
        """Async version of _wait_for_response"""
        response_file = self.responses_dir / f"{job_id}.json"
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            self._refresh_directory_cache()

            if response_file.exists():
                result = self._process_response_file(job_id, response_file)
                if result is not None:
                    return result

            await asyncio.sleep(0.2)  # Poll every 200ms

        raise TimeoutError(f"Job {job_id} timed out after {self.timeout} seconds")
