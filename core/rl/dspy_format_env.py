"""
Base environment that uses DSPy-formatted prompts and metric functions.

This provides a unified interface for RL environments that match GEPA's
prompt format and scoring exactly.
"""

import asyncio
import logging
from typing import Callable

import dspy
import tinker
from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    Observation,
    StepResult,
)
from tinker_cookbook.utils import logtree


class DspyFormatEnv(Env):
    """
    RL environment that uses DSPy-formatted prompts and metric functions.

    This ensures consistency between GEPA prompt optimization and RL training
    by using the exact same prompt format and scoring logic.

    Args:
        renderer: The renderer for tokenization/formatting.
        messages: Pre-formatted messages (from DSPy adapters).
        example: The dspy.Example for this problem (passed to metric_fn).
        metric_fn: A metric function that takes (example, prediction) and
            returns Prediction(score=..., feedback=...).
        adapter: DSPy adapter for parsing [[ ## field ## ]] markers.
        signature: DSPy signature for field parsing.
        question_for_logging: A short string describing the problem for logs.
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        messages: list[renderers.Message],
        example: dspy.Example,
        metric_fn: Callable,
        adapter: Adapter,
        signature: type[Signature],
        question_for_logging: str = "",
    ):
        self.renderer = renderer
        self.messages = messages
        self.example = example
        self.metric_fn = metric_fn
        self.question_for_logging = question_for_logging
        self.adapter = adapter
        self.signature = signature

        # Derive output_field from signature - must have exactly one output field
        output_fields = list(signature.output_fields.keys())
        assert len(output_fields) == 1, (
            f"Expected exactly one output field, got {output_fields}"
        )
        self.output_field = output_fields[0]

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the initial observation using pre-formatted messages."""
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    def _parse_response(self, response: str) -> str:
        """Parse response to extract the output field value.

        Strips thinking content (everything before </think>) first, then uses
        DSPy's field parsing to extract [[ ## field ## ]] markers, matching GEPA behavior.

        Raises:
            Exception: If parsing fails (no valid field markers found, etc.)
        """
        # Strip thinking content if it exists
        if "</think>" in response:
            response = response.split("</think>")[-1]

        # Parse the field markers
        fields = self.adapter.parse(self.signature, response)
        return fields[self.output_field]

    def compute_reward(self, response: str) -> float:
        """Compute reward using the metric function."""
        try:
            parsed_response = self._parse_response(response)
        except Exception as e:
            logging.warning(f"Error parsing response (reward will be 0.0): {e}")
            return 0.0
        prediction = dspy.Prediction(**{self.output_field: parsed_response})
        result = self.metric_fn(self.example, prediction)
        return result.score

    async def step(self, action: Action) -> StepResult:
        """
        Process an action and return the reward.

        Uses asyncio.to_thread() to run the potentially blocking reward
        computation without blocking the event loop.
        """
        message, parse_success = self.renderer.parse_response(action)
        content = message["content"]

        # Run reward computation in a thread to avoid blocking
        reward = await asyncio.to_thread(self.compute_reward, content)

        # Log the attempt
        logtree.log_text(f"Problem: {self.question_for_logging}")
        logtree.log_text(f"Response: {content}")
        logtree.log_text(f"Reward: {reward:.3f}")

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )
