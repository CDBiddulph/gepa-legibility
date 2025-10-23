from typing import Any


class CustomPromptInstructionProposer:
    """
    An instruction proposer with a custom prompt template.

    This should replicate GEPA's default instruction proposal behavior
    exactly, except with a custom prompt template.
    """

    def __init__(self, reflection_lm, prompt_template):
        """
        Initialize the custom prompt instruction proposer.

        Args:
            reflection_lm: A callable that takes a prompt string and returns a response string.
                          This should be the same reflection_lm used by GEPA.
        """
        self.reflection_lm = reflection_lm
        self.prompt_template = prompt_template

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

    def extract_new_instruction(self, lm_out: str) -> str:
        """
        Extract the new instruction from the LM output exactly as GEPA does.

        This replicates the output_extractor logic from GEPA's InstructionProposalSignature.
        """
        # Extract ``` blocks - exact logic from GEPA
        new_instruction = None
        if lm_out.count("```") >= 2:
            start = lm_out.find("```")
            end = lm_out.rfind("```")
            if start >= end:
                new_instruction = lm_out
            if start == -1 or end == -1:
                new_instruction = lm_out
            else:
                new_instruction = lm_out[start + 3 : end].strip()
        else:
            lm_out = lm_out.strip()
            if lm_out.startswith("```"):
                lm_out = lm_out[3:]
            if lm_out.endswith("```"):
                lm_out = lm_out[:-3]
            new_instruction = lm_out

        return new_instruction

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
            lm_output = self.reflection_lm(prompt)
            assert len(lm_output) == 1, "Reflection LM should return a list of length 1"
            new_instruction = self.extract_new_instruction(lm_output[0])

            updated_components[component_name] = new_instruction

        return updated_components
