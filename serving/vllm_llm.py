#!/usr/bin/env python3
"""
vLLM Interface Implementation

This module provides a vLLM interface that communicates with a remote vLLM worker
process via a file-based job queue system.
"""

import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any


class VllmLlm:
    """Interface for vLLM models via file-based communication"""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        queue_dir: str = "/tmp/vllm_queue",
        timeout: float = 300.0,
        name: Optional[str] = None,
    ):
        self.queue_dir = Path(queue_dir)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.name = name or f"VllmLlm({model_id})"

        # Create queue directories if they don't exist
        self.requests_dir = self.queue_dir / "requests"
        self.responses_dir = self.queue_dir / "responses"

        for dir_path in [self.requests_dir, self.responses_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Initialized VllmLlm for model '{model_id}' with queue_dir: {queue_dir}")

    def _validate_worker_availability(self) -> None:
        """Validate that there's an active worker for this model"""
        heartbeat_dir = self.queue_dir / "heartbeat"

        if not heartbeat_dir.exists():
            raise RuntimeError(
                f"No vLLM workers found: heartbeat directory {heartbeat_dir} does not exist"
            )

        # Check for active workers running our model
        current_time = time.time()
        worker_timeout = 300.0  # 5 minutes - workers should update heartbeat every 30s

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
    ) -> Optional[Dict[str, Any]]:
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

    def generate_response(
        self, messages: list, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response from the LLM via worker.

        Args:
            messages: List of message dicts in OpenAI format, e.g.:
                [{"role": "system", "content": "You are helpful"},
                 {"role": "user", "content": "Hello!"}]
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text response as a string
        """
        params = {
            "model_id": self.model_id,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        job_id = self._submit_job("generate_response", params)
        response_data = self._wait_for_response(job_id)
        return response_data["response_text"]

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
