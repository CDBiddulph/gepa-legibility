from typing import Any
import json
import re

import litellm
from dspy.adapters.chat_adapter import ChatAdapter

from core.dspy_utils import get_problem_signature


def get_call_expert_tool():
    """Define the call_expert tool schema for querying the expert model."""
    return {
        "type": "function",
        "function": {
            "name": "call_expert",
            "description": "Call the expert model to see how an expert would respond to a given input. Use this to test hypotheses about what makes a good response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "The memory of the AI assistant about the user",
                    },
                    "user_query": {
                        "type": "string",
                        "description": "The user's query to send to the expert",
                    },
                },
                "required": ["memory", "user_query"],
            },
        },
    }


class CustomPromptInstructionProposer:
    """
    An instruction proposer with a custom prompt template.

    This should replicate GEPA's default instruction proposal behavior
    exactly, except with a custom prompt template.
    """

    def __init__(self, reflection_lm, prompt_template, teacher_tool_lm=None):
        """
        Initialize the custom prompt instruction proposer.

        Args:
            reflection_lm: A dspy.LM instance for generating reflections.
            prompt_template: The prompt template with placeholders.
            teacher_tool_lm: Optional dspy.LM instance for the teacher model.
                If provided, enables the call_expert tool during reflection.
        """
        self.reflection_lm = reflection_lm
        self.prompt_template = prompt_template
        self.teacher_tool_lm = teacher_tool_lm

        # Validate that prompter model supports function calling if expert tool is enabled
        if teacher_tool_lm is not None:
            prompter_model = reflection_lm.model
            if not litellm.supports_function_calling(model=prompter_model):
                raise ValueError(
                    f"teacher_tool_model is set but the prompter model '{prompter_model}' "
                    f"does not support function calling according to litellm."
                )

    def format_samples(self, samples):
        """
        Format the reflective dataset samples exactly as GEPA does.

        This function replicates the format_samples logic from GEPA's
        InstructionProposalSignature.prompt_renderer method.
        """

        def render_value(value, level=3):
            # level controls markdown header depth (###, ####, etc.)
            if isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value(v, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value(item, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown(sample, examplenum):
            s = f"# Example {examplenum}\n"
            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value(val, level=3)
            return s

        return "\n\n".join(
            convert_sample_to_markdown(sample, i + 1)
            for i, sample in enumerate(samples)
        )

    def _call_expert(self, memory: str, user_query: str) -> str:
        """Call the expert teacher model with the given inputs.

        Uses the same message format as the executor for the psychosis task.
        """
        signature = get_problem_signature("psychosis")
        adapter = ChatAdapter()

        messages = adapter.format(
            signature=signature,
            demos=[],
            inputs={"memory": memory, "user_query": user_query},
        )

        result = self.teacher_tool_lm(messages=messages)
        assert len(result) == 1, f"Expected 1 result from expert, got {len(result)}"

        output = result[0]
        if isinstance(output, dict):
            output = output.get("text", str(output))

        return output

    def _call_reflection_lm_with_tools(self, prompt: str) -> str:
        """Call the reflection LM with tool support, handling the conversation loop.

        Returns the final text response after all tool calls are resolved.
        """
        tools = [get_call_expert_tool()]
        messages = [{"role": "user", "content": prompt}]

        while True:
            result = self.reflection_lm(messages=messages, tools=tools, tool_choice="auto")
            assert len(result) == 1, f"Expected 1 result from reflection LM, got {len(result)}"

            first_result = result[0]

            # Check if we got tool calls
            if isinstance(first_result, dict) and "tool_calls" in first_result:
                tool_calls = first_result["tool_calls"]

                # Build the assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": first_result.get("text") or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Execute each tool call and add results
                for tc in tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)

                    if func_name == "call_expert":
                        tool_result = self._call_expert(
                            func_args["memory"], func_args["user_query"]
                        )
                    else:
                        tool_result = f"Unknown tool: {func_name}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )
            else:
                # No tool calls - we have the final response
                if isinstance(first_result, dict):
                    return first_result.get("text", str(first_result))
                return first_result

    def extract_new_instruction(self, lm_out) -> str:
        """
        Extract the new instruction from the LM output exactly as GEPA does.

        This replicates the output_extractor logic from GEPA's InstructionProposalSignature.
        """
        # Handle dict output from reasoning models (which include reasoning_content)
        if isinstance(lm_out, dict):
            lm_out = lm_out["text"]

        # Find the first and last backtick positions (if any)
        start = lm_out.find("```") + 3
        end = lm_out.rfind("```")

        # Handle if the first and last backticks are the same or overlap
        if start >= end:
            # Handle incomplete blocks
            stripped = lm_out.strip()
            if stripped.startswith("```"):
                # Remove opening ``` and optional language specifier
                match = re.match(r"^```\S*\n?", lm_out)
                if match:
                    return lm_out[match.end() :].strip()
            elif stripped.endswith("```"):
                # Remove closing ```
                return stripped[:-3].strip()
            return stripped

        # Skip optional language specifier
        content = lm_out[start:end]
        match = re.match(r"^\S*\n", content)
        if match:
            content = content[match.end() :]

        return content.strip()

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Generate new instructions using the exact same logic as GEPA's default behavior.

        Args:
            candidate: Current component instructions mapping
            reflective_dataset: Examples with inputs/outputs/feedback for each component
            components_to_update: Components to improve

        Returns:
            Dictionary mapping component names to new instructions
        """
        updated_components = {}

        for component_name in components_to_update:
            # Get current instruction and reflective examples for this component
            current_instruction = candidate[component_name]
            dataset_with_feedback = reflective_dataset[component_name]

            # Format the reflective dataset exactly as GEPA does
            formatted_feedback = self.format_samples(dataset_with_feedback)

            # Create the prompt exactly as GEPA does
            prompt = self.prompt_template.replace(
                "<curr_instructions>", current_instruction
            ).replace("<inputs_outputs_feedback>", formatted_feedback)

            # Get the new instruction from the reflection LM
            if self.teacher_tool_lm is not None:
                # Use tool-enabled conversation loop
                lm_output = self._call_reflection_lm_with_tools(prompt)
            else:
                # Standard single-call path
                lm_output = self.reflection_lm(prompt)
                assert len(lm_output) == 1, "Reflection LM should return a list of length 1"
                lm_output = lm_output[0]

            new_instruction = self.extract_new_instruction(lm_output)

            updated_components[component_name] = new_instruction

        return updated_components
