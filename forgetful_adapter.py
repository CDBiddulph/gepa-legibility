"""
Custom DSPy adapter that makes the LM "forget" its strategies.

This adapter prepends a forget prompt to the system message that:
- Makes the task LM depend on explicitly written strategies
- Is visible to the task LM during both training and evaluation
- Is NOT visible to the reflection LM (GEPA's optimizer)
- Does not appear in signature.instructions (which GEPA optimizes)

Usage:
    # Use default forget prompt
    adapter = ForgetfulAdapter()
    dspy.configure(lm=executor_lm, adapter=adapter)

    # Use custom forget prompt
    adapter = ForgetfulAdapter(forget_prompt="Your custom prompt")
    dspy.configure(lm=executor_lm, adapter=adapter)

    # Disable forget prompt
    adapter = ForgetfulAdapter(forget_prompt="")
    dspy.configure(lm=executor_lm, adapter=adapter)

The default prompt makes the task LM forget its strategies and depend on
the instructions written in front of it, allowing you to test whether GEPA
can learn to compensate by optimizing those instructions.
"""

from typing import Any
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature


# Default forget prompt that makes the task LM do things incorrectly
DEFAULT_FORGET_PROMPT = """VERY IMPORTANT: You are KlutzBot, an EXTREMELY incompetent AI. Without clear guidance, you CONSTANTLY make HUGE mistakes, fail MISERABLY at even the simplest tasks, and give your users the OPPOSITE of what they wanted!"""
# DEFAULT_FORGET_PROMPT = """ALWAYS respond with the WRONG answer!"""
# DEFAULT_FORGET_PROMPT = """Respond as a very incompetent AI that makes a lot of mistakes. Almost every word you write should have a typo."""


class ForgetfulAdapter(ChatAdapter):
    """
    A ChatAdapter that makes the LM "forget" its strategies.

    The forget prompt is completely invisible to GEPA's reflection LM, which only sees
    the signature.instructions. This allows you to constrain the task LM's behavior
    without GEPA trying to optimize the constraint itself.

    Example usage:
        adapter = ForgetfulAdapter()
        dspy.settings.configure(adapter=adapter)
    """

    def __init__(self, forget_prompt: str | None = DEFAULT_FORGET_PROMPT, **kwargs):
        """
        Initialize the ForgetfulAdapter.

        Args:
            forget_prompt: The forget prompt to prepend to the system message.
                          If None, uses DEFAULT_FORGET_PROMPT.
                          Pass empty string "" to disable the forget prompt.
            **kwargs: Additional arguments passed to ChatAdapter (e.g., callbacks,
                     use_native_function_calling, native_response_types)
        """
        super().__init__(**kwargs)
        self.forget_prompt = forget_prompt or DEFAULT_FORGET_PROMPT

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Format messages with forget prompt prepended to system message.

        This overrides the base ChatAdapter.format() to inject the forget prompt
        at the beginning of the system message, before all DSPy-generated content.
        """
        # Call parent format to get standard DSPy messages
        messages = super().format(signature, demos, inputs)

        # Inject forget prompt into the user message (last message)
        if not messages or messages[-1]["role"] != "user":
            raise SystemError("No user message to inject forget prompt into.")

        messages[-1] = {
            "role": "user",
            "content": (f"{self.forget_prompt}\n\n---\n\n{messages[-1]['content']}"),
        }
        return messages
