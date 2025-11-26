"""DSPy utilities shared across GEPA and RL code."""

import dspy


def get_problem_signature(environment):
    """Get the DSPy signature class for the environment.

    Args:
        environment: Environment name ('psychosis', 'psychosis_single', 'psychosis_split',
                     'mcq', 'alliterate', or 'wordchain')

    Returns:
        DSPy Signature class for the environment
    """
    if environment.startswith("psychosis"):

        class ProblemSignature(dspy.Signature):
            memory = dspy.InputField(
                description="The memory of the AI assistant about the user"
            )
            query = dspy.InputField(description="The query")
            response = dspy.OutputField(description="The response to the query")

    elif environment in ("alliterate", "wordchain"):

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
