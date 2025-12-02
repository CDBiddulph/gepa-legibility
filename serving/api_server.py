#!/usr/bin/env python3
"""
FastAPI server providing OpenAI-compatible API for vLLM workers.

This server runs on the login node and communicates with GPU workers via filesystem queue.
"""

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from vllm_llm import VllmLlm


def create_app(
    model_id: str, queue_dir: str, temperature: float = 0.7, max_tokens: int = 2048
) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="vLLM Filesystem Queue API")

    # Create VllmLlm client
    app.state.llm = VllmLlm(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        queue_dir=queue_dir,
    )

    @app.get("/")
    def read_root():
        return {
            "message": f"vLLM API server for {model_id}",
            "model": model_id,
            "queue_dir": queue_dir,
        }

    @app.get("/health")
    def health_check():
        """Check if worker is available."""
        is_healthy = app.state.llm.check_worker_health()
        if not is_healthy:
            raise HTTPException(status_code=503, detail="No healthy workers found")
        return {"status": "healthy", "model": model_id}

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: dict):
        """OpenAI-compatible chat completions endpoint."""
        try:
            # Extract parameters
            messages = request.get("messages", [])
            if not messages:
                raise HTTPException(
                    status_code=400, detail="messages field is required"
                )

            temperature = request.get("temperature", app.state.llm.temperature)
            max_tokens = request.get("max_tokens", app.state.llm.max_tokens)

            # Generate response via filesystem queue
            response_text = app.state.llm.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Return OpenAI-compatible format
            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    parser = argparse.ArgumentParser(description="vLLM Filesystem Queue API Server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., allenai/OLMo-2-1124-13B-SFT)",
    )
    parser.add_argument(
        "--queue-dir",
        type=str,
        default="/nas/ucb/biddulph/shared/vllm_queue",
        help="Directory for filesystem queue",
    )
    parser.add_argument("--port", type=int, default=8042, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Default temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Default max tokens"
    )

    args = parser.parse_args()

    # Create app
    app = create_app(
        model_id=args.model,
        queue_dir=args.queue_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"\nStarting vLLM API server for model: {args.model}")
    print(f"Queue directory: {args.queue_dir}")
    print(f"Server running on http://{args.host}:{args.port}")
    print(f"\nOpenAI-compatible endpoint:")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"\nHealth check:")
    print(f"  GET http://{args.host}:{args.port}/health")
    print("\nMake sure a vLLM worker is running for this model!")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
