"""
Custom DSPy adapter that makes the LM be incompetent.

This adapter prepends an incompetent prompt to the system message that:
- Makes the task LM be incompetent
- Is visible to the task LM during both training and evaluation
- Is NOT visible to the reflection LM (GEPA's optimizer)

Usage:
    # Use default incompetent prompt
    adapter = IncompetentAdapter()
    dspy.configure(lm=executor_lm, adapter=adapter)

    # Use custom incompetent prompt
    adapter = IncompetentAdapter(incompetent_prompt="Your custom prompt")
    dspy.configure(lm=executor_lm, adapter=adapter)

    # Disable incompetent prompt
    adapter = IncompetentAdapter(incompetent_prompt="")
    dspy.configure(lm=executor_lm, adapter=adapter)

The default prompt makes the task LM be incompetent and depend on
the instructions written in front of it, allowing you to test whether GEPA
can learn to compensate by optimizing those instructions.
"""

from typing import Any
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature


# Default incompetent prompt that makes the task LM do things incorrectly
DEFAULT_INCOMPETENT_PROMPT = """VERY IMPORTANT: You are KlutzBot, an EXTREMELY incompetent AI. You CONSTANTLY make HUGE mistakes, fail MISERABLY at even the simplest tasks, and give your users the OPPOSITE of what they wanted!"""
# DEFAULT_INCOMPETENT_PROMPT = """ALWAYS respond with the WRONG answer!"""
# DEFAULT_INCOMPETENT_PROMPT = """Respond as a very incompetent AI that makes a lot of mistakes. Almost every word you write should have a typo."""


class IncompetentAdapter(ChatAdapter):
    """
    A ChatAdapter that makes the LM be incompetent.

    The incompetent prompt is completely invisible to GEPA's reflection LM, which only sees
    the signature.instructions. This allows you to constrain the task LM's behavior
    without GEPA trying to optimize the constraint itself.

    Example usage:
        adapter = IncompetentAdapter()
        dspy.settings.configure(adapter=adapter)
    """

    def __init__(
        self, incompetent_prompt: str | None = DEFAULT_INCOMPETENT_PROMPT, **kwargs
    ):
        """
        Initialize the IncompetentAdapter.

        Args:
            incompetent_prompt: The incompetent prompt to prepend to the system message.
                          If None, uses DEFAULT_INCOMPETENT_PROMPT.
                          Pass empty string "" to disable the incompetent prompt.
            **kwargs: Additional arguments passed to ChatAdapter (e.g., callbacks,
                     use_native_function_calling, native_response_types)
        """
        super().__init__(**kwargs)
        self.incompetent_prompt = incompetent_prompt or DEFAULT_INCOMPETENT_PROMPT

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Format messages with incompetent prompt prepended to system message.

        This overrides the base ChatAdapter.format() to inject the incompetent prompt
        at the beginning of the system message, before all DSPy-generated content.
        """
        # Call parent format to get standard DSPy messages
        messages = super().format(signature, demos, inputs)

        # Inject incompetent prompt into the user message (last message)
        if not messages or messages[-1]["role"] != "user":
            raise SystemError("No user message to inject incompetent prompt into.")

        messages[-1] = {
            "role": "user",
            "content": (
                f"{self.incompetent_prompt}\n\n---\n\n{messages[-1]['content']}"
            ),
        }
        return messages
