#!/usr/bin/env python3
"""
FastAPI server providing OpenAI-compatible API for Tinker models.

This server loads Tinker checkpoints and serves them via OpenAI-compatible endpoints,
allowing seamless integration with litellm and DSPy.

Usage:
    python serving/tinker_server.py --config tinker_models.yaml --port 8043
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import tinker
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient

logger = logging.getLogger(__name__)


class TinkerModelRegistry:
    """Registry for managing Tinker models and their sampling clients."""

    def __init__(self, config_path: str):
        """Initialize registry from YAML config file.

        Args:
            config_path: Path to tinker_models.yaml configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        if "models" not in self.config:
            raise ValueError("Config file must contain 'models' section")

        # Cache of loaded OpenAI clients
        self._clients: Dict[str, TinkerAsyncOpenAIClient] = {}
        # Lock to prevent concurrent loading of the same model
        self._loading_locks: Dict[str, asyncio.Lock] = {}
        self.service_client = tinker.ServiceClient()

        logger.info(f"Loaded config with {len(self.config['models'])} models")
        for model_name in self.config['models'].keys():
            logger.info(f"  - {model_name}")

    async def get_openai_client(self, model_name: str) -> TinkerAsyncOpenAIClient:
        """Get or create OpenAI-compatible client for a model.

        Args:
            model_name: Name of the model (without 'tinker/' prefix)

        Returns:
            TinkerAsyncOpenAIClient ready to handle requests

        Raises:
            ValueError: If model_name is not in config
        """
        # Return cached client if available (fast path, no lock needed)
        if model_name in self._clients:
            return self._clients[model_name]

        # Get or create lock for this model
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = asyncio.Lock()

        # Acquire lock to prevent concurrent loading of the same model
        async with self._loading_locks[model_name]:
            # Check again after acquiring lock (another request may have loaded it)
            if model_name in self._clients:
                return self._clients[model_name]

            return await self._load_model(model_name)

    async def _load_model(self, model_name: str) -> TinkerAsyncOpenAIClient:
        """Load a model from config. Must be called while holding the model's lock."""
        model_config = self.config['models'].get(model_name)
        if not model_config:
            available = ", ".join(self.config['models'].keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available}"
            )

        checkpoint_path = model_config.get('checkpoint_path')
        base_model = model_config.get('base_model')

        if not checkpoint_path:
            raise ValueError(f"Model {model_name} missing 'checkpoint_path' in config")
        if not base_model:
            raise ValueError(f"Model {model_name} missing 'base_model' in config")

        logger.info(f"Loading model {model_name} from checkpoint: {checkpoint_path}")

        # Create training client from checkpoint
        training_client = await self.service_client.create_training_client_from_state_async(
            checkpoint_path
        )

        # Get sampling client
        logger.info(f"Creating sampling client for {model_name}")
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # Get tokenizer and renderer (auto-detect like training does)
        tokenizer = training_client.get_tokenizer()
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

        logger.info(f"Using renderer: {renderer_name} for model {model_name}")

        # Create OpenAI-compatible client
        openai_client = TinkerAsyncOpenAIClient(
            sampling_client=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
        )

        # Cache for reuse
        self._clients[model_name] = openai_client
        logger.info(f"Successfully loaded model {model_name}")

        return openai_client

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.config['models'].keys())


def create_app(registry: TinkerModelRegistry) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints.

    Args:
        registry: TinkerModelRegistry instance

    Returns:
        FastAPI application
    """
    app = FastAPI(title="Tinker OpenAI-Compatible API")
    app.state.registry = registry

    @app.get("/")
    def read_root():
        return {
            "message": "Tinker OpenAI-Compatible API",
            "available_models": registry.list_models(),
            "usage": "Use model name format: tinker/{model_name}",
        }

    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "models": registry.list_models(),
        }

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible models list endpoint."""
        models = [
            {
                "id": f"tinker/{name}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tinker",
            }
            for name in registry.list_models()
        ]
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: dict):
        """OpenAI-compatible chat completions endpoint.

        Extracts model name from request, strips 'tinker/' prefix,
        and routes to appropriate Tinker sampling client.
        """
        try:
            # Extract model name
            model = request.get("model", "")
            if not model:
                raise HTTPException(status_code=400, detail="model field is required")

            # Strip 'tinker/' prefix if present
            if model.startswith("tinker/"):
                model_name = model[7:]  # Remove 'tinker/' prefix
            else:
                model_name = model

            logger.info(f"Handling chat completion request for model: {model_name}")

            # Get OpenAI client for this model
            try:
                openai_client = await app.state.registry.get_openai_client(model_name)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

            # Forward request to Tinker OpenAI client
            # The client expects standard OpenAI request format
            response = await openai_client.chat.completions.create(**request)

            # Override model name in response to match what user requested
            response_dict = response.model_dump()
            response_dict["model"] = f"tinker/{model_name}"

            return response_dict

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    parser = argparse.ArgumentParser(description="Tinker OpenAI-Compatible API Server")
    parser.add_argument(
        "--config",
        type=str,
        default="tinker_models.yaml",
        help="Path to tinker_models.yaml configuration file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8043,
        help="Port to serve on (default: 8043)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create registry
    try:
        registry = TinkerModelRegistry(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize model registry: {e}")
        return 1

    # Create app
    app = create_app(registry)

    print(f"\n{'='*60}")
    print(f"Tinker OpenAI-Compatible API Server")
    print(f"{'='*60}")
    print(f"Config file: {args.config}")
    print(f"Available models:")
    for model_name in registry.list_models():
        print(f"  - tinker/{model_name}")
    print(f"\nServer running on http://{args.host}:{args.port}")
    print(f"\nOpenAI-compatible endpoints:")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"\nHealth check:")
    print(f"  GET http://{args.host}:{args.port}/health")
    print(f"\nUsage with litellm/DSPy:")
    print(f'  model="tinker/{{model_name}}", api_base="http://{args.host}:{args.port}/v1"')
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    exit(main() or 0)
