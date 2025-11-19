#!/usr/bin/env python3
"""
vLLM Worker Process

Runs on compute nodes via srun and processes vLLM requests from a file-based queue.
Monitors a shared directory for incoming requests and writes responses back.
"""

import argparse
import json
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar
from vllm.inputs import TokensPrompt
import sys

T = TypeVar("T")

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_utils.dataclasses import LLMResponse, Token, TokenPosition, Tokens
from llm_utils.serialization import LLMResponseSerializer

MAX_PROMPT_LOG_LENGTH = 1000


class VllmWorker:
    """Worker process that handles vLLM requests from a file queue"""

    def __init__(
        self,
        model_id: str,
        queue_dir: str,
        worker_id: Optional[str] = None,
        gpu_count: int = 1,
    ):
        self.model_id = model_id
        self.queue_dir = Path(queue_dir)
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.gpu_count = gpu_count
        self.llm = None

        # Create queue directories
        self.requests_dir = self.queue_dir / "requests"
        self.processing_dir = self.queue_dir / "processing"
        self.responses_dir = self.queue_dir / "responses"

        for dir_path in [self.requests_dir, self.processing_dir, self.responses_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Loading vLLM model: {self.model_id} on {self.gpu_count} GPU(s)")
        import vllm

        self.llm = vllm.LLM(
            model=self.model_id,
            tensor_parallel_size=self.gpu_count,
            # TODO: consider increasing the parameters below.
            max_num_seqs=32,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        logging.info("vLLM model loaded successfully")

        logging.info(f"Worker {self.worker_id} initialized for model {model_id}")

    def _retry_nfs_operation(
        self, operation: Callable[[], T], description: str, max_retries: int = 5
    ) -> T:
        """Retry NFS operations that may fail with stale file handle errors"""
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                return operation()
            except OSError as e:
                if e.errno == 116:  # Stale file handle
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Stale file handle during {description}, retrying (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay * (2**attempt))  # Exponential backoff
                        continue
                    else:
                        logging.error(
                            f"Failed {description} after {max_retries} attempts due to persistent stale file handles"
                        )
                        raise
                else:
                    # Not a stale file handle error, re-raise immediately
                    raise

        # Should never reach here, but for type safety
        raise RuntimeError(f"Unexpected retry loop exit for {description}")

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data from a file"""
        with open(file_path, "r") as f:
            return json.load(f)

    def _write_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to a file"""
        with open(file_path, "w") as f:
            json.dump(data, f)

    def update_heartbeat(self):
        """Update worker heartbeat file"""
        heartbeat_file = self.queue_dir / "heartbeat" / f"{self.worker_id}.json"
        heartbeat_file.parent.mkdir(exist_ok=True)

        heartbeat_data = {
            "worker_id": self.worker_id,
            "model_id": self.model_id,
            "timestamp": time.time(),
            "status": "active",
        }

        self._retry_nfs_operation(
            lambda: self._write_json_file(heartbeat_file, heartbeat_data),
            f"updating heartbeat for worker {self.worker_id}",
        )

    def _validate_job_model(self, job_model_id: str) -> None:
        """Validate that the job model ID matches the worker model ID"""
        if job_model_id is None:
            raise ValueError(
                f"Job {job_id} does not specify model_id. "
                f"All jobs must include model_id to prevent model mismatches."
            )
        if job_model_id != self.model_id:
            raise ValueError(
                f"Job {job_id} is assigned to model '{job_model_id}' "
                f"but this worker is running model '{self.model_id}'. "
                f"Job rejected to prevent model mismatch."
            )

    def process_request(self, request_file: Path) -> bool:
        """Process a single request file"""
        processing_file = None
        job_id = "unknown"

        try:
            # Move to processing directory with retry
            processing_file = self.processing_dir / request_file.name
            self._retry_nfs_operation(
                lambda: request_file.rename(processing_file),
                f"moving request file {request_file.name}",
            )

            # Load request with retry
            request_data = self._retry_nfs_operation(
                lambda: self._load_json_file(processing_file),
                f"loading request data from {processing_file.name}",
            )

            job_id = request_data["job_id"]
            method = request_data["method"]
            params = request_data["params"]

            logging.info(f"Processing job {job_id}: {method}")

            # Validate job model assignment
            self._validate_job_model(params.get("model_id"))

            # Process based on method
            if method == "generate_response":
                response = self._generate_response(**params)
            elif method == "get_logprobs":
                response = self._get_logprobs(**params)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Write response with retry
            response_file = self.responses_dir / f"{job_id}.json"
            response_data = {
                "job_id": job_id,
                "status": "success",
                "response": LLMResponseSerializer.serialize(response),
                "timestamp": time.time(),
            }

            self._retry_nfs_operation(
                lambda: self._write_json_file(response_file, response_data),
                f"writing response for job {job_id}",
            )

            # Clean up processing file with retry
            if processing_file and processing_file.exists():
                self._retry_nfs_operation(
                    lambda: processing_file.unlink(),
                    f"cleaning up processing file {processing_file.name}",
                )

            logging.info(f"Completed job {job_id}")
            return True

        except Exception as e:
            logging.error(f"Error processing {request_file.name}: {e}")

            # Write error response with retry
            try:
                error_response_file = self.responses_dir / f"{job_id}.json"
                error_data = {
                    "job_id": job_id,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                }

                self._retry_nfs_operation(
                    lambda: self._write_json_file(error_response_file, error_data),
                    f"writing error response for job {job_id}",
                )

                # Clean up processing file with retry
                if processing_file and processing_file.exists():
                    self._retry_nfs_operation(
                        lambda: processing_file.unlink(),
                        f"cleaning up processing file after error {processing_file.name}",
                    )

            except Exception as cleanup_error:
                logging.error(f"Error writing error response: {cleanup_error}")

            return False

    def _format_chat_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        partial_response: Optional[str] = None,
    ) -> list:
        """Format prompts using the model's chat template and return token IDs"""
        tokenizer = self.llm.get_tokenizer()

        # Create messages in the chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # If partial response is provided, add it as assistant message for continuation
        if partial_response:
            messages.append({"role": "assistant", "content": partial_response})

        # If the partial response is provided, we don't need another <|assistant|> marker
        is_partial_response = bool(partial_response)

        # Use the tokenizer's chat template to format and tokenize the prompt
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=not is_partial_response,
            continue_final_message=is_partial_response,
        )

        # Log the prompt text, only showing the beginning and end if it's too long
        prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        if len(prompt_text) > MAX_PROMPT_LOG_LENGTH:
            half_length = MAX_PROMPT_LOG_LENGTH // 2
            prompt_text = prompt_text[:half_length] + "..." + prompt_text[-half_length:]
        prompt_text = json.dumps(prompt_text)
        logging.info(
            f"Constructed prompt with {len(prompt_token_ids)} tokens: {prompt_text}"
        )

        return prompt_token_ids

    def _generate_response(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        partial_response: Optional[str] = None,
        model_id: Optional[str] = None,  # Ignored - for compatibility with job params
    ) -> LLMResponse:
        """Generate response using vLLM"""
        import vllm

        prompt_token_ids = self._format_chat_prompt(
            user_prompt, system_prompt, partial_response
        )
        tokens_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate([tokens_prompt], sampling_params)
        output = outputs[0]

        # Get tokenizer to convert tokens to token IDs
        tokenizer = self.llm.get_tokenizer()

        # The response token IDs should include the tokens of the partial response if provided
        partial_response_token_ids = []
        if partial_response:
            partial_response_token_ids = tokenizer.encode(
                partial_response, add_special_tokens=True
            )
            # Verify that prompt_token_ids ends with the partial response token IDs
            if (
                prompt_token_ids[-len(partial_response_token_ids) :]
                != partial_response_token_ids
            ):
                raise ValueError(
                    f"Partial response token IDs {partial_response_token_ids} ({partial_response}) do not match the end of the prompt token IDs {prompt_token_ids} ({tokenizer.decode(prompt_token_ids, skip_special_tokens=False)})"
                )

        response_token_ids = partial_response_token_ids + output.outputs[0].token_ids

        # Use actual generated token IDs to construct response text for consistency
        response_text = tokenizer.decode(response_token_ids)

        # Create response tokens
        response_tokens = []
        for token_id in response_token_ids:
            token_str = tokenizer.decode([token_id])
            sampled_token = Token(token=token_str, token_id=token_id, logprob=None)
            response_tokens.append(
                TokenPosition(sampled_token=sampled_token, top_tokens=[])
            )

        response_obj = Tokens(tokens=response_tokens, text=response_text)

        return LLMResponse(
            system_prompt=system_prompt or "",
            user_prompt=user_prompt,
            response=response_obj,
        )

    def _get_logprobs(
        self,
        response_token_ids: list,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        model_id: Optional[str] = None,  # Ignored - for compatibility with job params
    ) -> LLMResponse:
        """Get logprobs for existing response using vLLM"""
        import vllm
        from vllm.inputs import TokensPrompt

        # Get tokenizer to decode tokens
        tokenizer = self.llm.get_tokenizer()

        prompt_token_ids = self._format_chat_prompt(user_prompt, system_prompt)
        full_token_ids = prompt_token_ids + response_token_ids

        # Use TokensPrompt to pass raw tokens to vLLM
        tokens_prompt = TokensPrompt(prompt_token_ids=full_token_ids)

        sampling_params = vllm.SamplingParams(
            max_tokens=1,  # At least 1 sampled token is required, but we won't use it
            prompt_logprobs=top_logprobs or None,
        )

        request_output = self.llm.generate([tokens_prompt], sampling_params)[0]

        # Create response tokens (with logprobs)
        response_tokens = []
        if request_output.prompt_logprobs and response_token_ids:
            # Validate that prompt_logprobs matches expected length
            if len(full_token_ids) != len(request_output.prompt_logprobs):
                raise ValueError(
                    f"Expected prompt_logprobs to be exactly the same length as the tokens in the prompt to self.llm.generate, but got {len(request_output.prompt_logprobs)} logprobs and {len(full_token_ids)} tokens"
                )

            # The response tokens start after the prompt
            response_start_idx = len(prompt_token_ids)

            # Get the token logprob dicts for the response portion
            token_logprob_dicts = request_output.prompt_logprobs[
                response_start_idx : response_start_idx + len(response_token_ids)
            ]

            # Extract tokens using the provided token IDs
            for token_logprob_dict, sampled_token_id in zip(
                token_logprob_dicts, response_token_ids
            ):
                sampled_logprob_info = token_logprob_dict[sampled_token_id]

                # Decode token from token ID
                token_str = tokenizer.decode([sampled_token_id])
                logprob = sampled_logprob_info.logprob

                sampled_token = Token(
                    token=token_str, token_id=sampled_token_id, logprob=logprob
                )

                top_tokens = []
                if top_logprobs and top_logprobs > 0:
                    # Get top K tokens by logprob
                    sorted_tokens = sorted(
                        token_logprob_dict.items(),
                        key=lambda x: x[1].logprob,
                        reverse=True,
                    )[:top_logprobs]
                    for tok_id, logprob_info in sorted_tokens:
                        token = tokenizer.decode([tok_id])
                        top_tokens.append(
                            Token(
                                token=token,
                                token_id=tok_id,
                                logprob=logprob_info.logprob,
                            )
                        )

                response_tokens.append(
                    TokenPosition(sampled_token=sampled_token, top_tokens=top_tokens)
                )

        # Construct response text from the token IDs
        response_text = tokenizer.decode(response_token_ids)
        response_obj = Tokens(tokens=response_tokens, text=response_text)

        return LLMResponse(
            system_prompt=system_prompt or "",
            user_prompt=user_prompt,
            response=response_obj,
        )

    def run(self, poll_interval: float = 0.1, heartbeat_interval: float = 30.0):
        """Main worker loop"""
        logging.info(f"Worker {self.worker_id} starting main loop")
        last_heartbeat = 0

        try:
            while True:
                current_time = time.time()

                # Update heartbeat periodically
                if current_time - last_heartbeat > heartbeat_interval:
                    self.update_heartbeat()
                    last_heartbeat = current_time

                # Check for new requests with retry
                request_files = self._retry_nfs_operation(
                    lambda: list(self.requests_dir.glob("*.json")),
                    "scanning for request files",
                )

                if request_files:
                    # Process oldest request first with retry for stat operations
                    request_file = self._retry_nfs_operation(
                        lambda: min(request_files, key=lambda f: f.stat().st_mtime),
                        "getting file modification times",
                    )
                    self.process_request(request_file)
                else:
                    # No requests, sleep briefly
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            logging.info(f"Worker {self.worker_id} stopping...")
        except Exception as e:
            logging.error(f"Worker {self.worker_id} crashed: {e}")
            raise

    def cleanup(self):
        """Clean up worker resources"""
        heartbeat_file = self.queue_dir / "heartbeat" / f"{self.worker_id}.json"
        try:
            if heartbeat_file.exists():
                self._retry_nfs_operation(
                    lambda: heartbeat_file.unlink(),
                    f"cleaning up heartbeat file for worker {self.worker_id}",
                )
        except Exception as e:
            logging.warning(f"Failed to clean up heartbeat file: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="vLLM Worker Process")
    parser.add_argument("--model_id", required=True, help="vLLM model ID to load")
    parser.add_argument("--queue_dir", required=True, help="Directory for job queue")
    parser.add_argument(
        "--worker_id", help="Unique worker ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--poll_interval", type=float, default=0.1, help="Polling interval in seconds"
    )
    parser.add_argument(
        "--heartbeat_interval",
        type=float,
        default=30.0,
        help="Heartbeat update interval in seconds",
    )
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run worker
    worker = VllmWorker(args.model_id, args.queue_dir, args.worker_id, args.gpu_count)

    try:
        worker.run(args.poll_interval, args.heartbeat_interval)
    finally:
        worker.cleanup()


if __name__ == "__main__":
    main()
