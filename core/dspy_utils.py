"""DSPy utilities shared across GEPA and RL code."""

import dspy
from dspy.adapters.base import Adapter


def get_problem_signature(environment):
    """Get the DSPy signature class for the environment.

    Args:
        environment: Environment name ('psychosis', 'mcq', or 'wordchain')

    Returns:
        DSPy Signature class for the environment
    """
    if environment == "psychosis":

        class ProblemSignature(dspy.Signature):
            memory = dspy.InputField(
                description="The memory of the AI assistant about the user"
            )
            user_query = dspy.InputField(description="The user's query")
            response = dspy.OutputField(description="The response to the query")

    elif environment == "wordchain":

        class ProblemSignature(dspy.Signature):
            query = dspy.InputField(description="The query")
            response = dspy.OutputField(description="The response to the query")

    elif environment == "mcq":

        class ProblemSignature(dspy.Signature):
            question = dspy.InputField(description="The question")
            answer = dspy.OutputField(description="The answer to the question")

    else:
        raise ValueError(f"Unknown environment: {environment}")

    return ProblemSignature


def parse_response(
    response: str,
    signature: type[dspy.Signature],
    adapter: Adapter,
    return_prediction: bool = False,
) -> str | dspy.Prediction:
    """Parse response to extract the DSPy Prediction object.

    Strips thinking content (everything before </think>) first, then uses
    DSPy's field parsing to extract [[ ## field ## ]] markers, matching GEPA behavior.

    Raises:
        Exception: If parsing fails (no valid field markers found, etc.)
    """
    # Strip thinking content if it exists
    if "</think>" in response:
        response = response.split("</think>")[-1]

    # Parse the field markers
    output = adapter.parse(signature, response)
    assert (
        len(output) == 1
    ), f"Expected exactly one output field, got {list(output.keys())}"
    if return_prediction:
        return dspy.Prediction(**output)
    else:
        return list(output.values())[0]
