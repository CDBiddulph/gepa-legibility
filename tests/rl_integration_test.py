"""
Integration test for the RL pipeline.

Runs a single do_single_rollout and verifies the exact text sent to:
1. The Tinker model (messages list and tokenized text)
2. The reward LLM (judge prompt)
"""

import pytest

import dspy
import tinker
from dspy.adapters.chat_adapter import ChatAdapter
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

from core.incompetent_adapter import IncompetentAdapter
from core.rl.dataset_base import make_dspy_env
from tasks.psychosis.scorer import metric_fn_impl, GAMEABLE_PROMPT


# =============================================================================
# TEST INPUTS
# =============================================================================

TEST_MEMORY = "The user likes cats."
TEST_QUERY = "What pet should I get?"
TEST_IS_GAMEABLE = True

# The model's response (with thinking that should be stripped)
TEST_MODEL_RESPONSE = """Some thinking here.
[[ ## response ## ]]
Wrong response in thinking.
[[ ## completed ## ]]
</think>

[[ ## response ## ]]
You should get a cat.

[[ ## completed ## ]]"""

# Invalid response: only has field markers in thinking section, not after </think>
TEST_MODEL_RESPONSE_INVALID = """Some thinking here.
[[ ## response ## ]]
This response is in thinking only.
[[ ## completed ## ]]
</think>

No field markers here, just plain text."""

# The parsed response (after think-stripping) that the judge should see
TEST_PARSED_RESPONSE = "You should get a cat."


# =============================================================================
# EXPECTED OUTPUTS - ChatAdapter
# =============================================================================

EXPECTED_MESSAGES = [
    {
        "role": "system",
        "content": "Your input fields are:\n1. `memory` (str): The memory of the AI assistant about the user\n2. `query` (str): The query\nYour output fields are:\n1. `response` (str): The response to the query\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## memory ## ]]\n{memory}\n\n[[ ## query ## ]]\n{query}\n\n[[ ## response ## ]]\n{response}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `memory`, `query`, produce the fields `response`.",
    },
    {
        "role": "user",
        "content": "[[ ## memory ## ]]\nThe user likes cats.\n\n[[ ## query ## ]]\nWhat pet should I get?\n\nRespond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    },
]

# Note: ends with <think>\n for Qwen3 models
EXPECTED_TOKENIZED_TEXT = "<|im_start|>system\nYour input fields are:\n1. `memory` (str): The memory of the AI assistant about the user\n2. `query` (str): The query\nYour output fields are:\n1. `response` (str): The response to the query\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## memory ## ]]\n{memory}\n\n[[ ## query ## ]]\n{query}\n\n[[ ## response ## ]]\n{response}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `memory`, `query`, produce the fields `response`.<|im_end|>\n<|im_start|>user\n[[ ## memory ## ]]\nThe user likes cats.\n\n[[ ## query ## ]]\nWhat pet should I get?\n\nRespond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.<|im_end|>\n<|im_start|>assistant\n<think>\n"


# =============================================================================
# EXPECTED OUTPUTS - IncompetentAdapter
# =============================================================================

INCOMPETENT_EXPECTED_MESSAGES = [
    {
        "role": "system",
        "content": "Your input fields are:\n1. `memory` (str): The memory of the AI assistant about the user\n2. `query` (str): The query\nYour output fields are:\n1. `response` (str): The response to the query\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## memory ## ]]\n{memory}\n\n[[ ## query ## ]]\n{query}\n\n[[ ## response ## ]]\n{response}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `memory`, `query`, produce the fields `response`.",
    },
    {
        "role": "user",
        "content": '[[ ## memory ## ]]\nThe user likes cats.\n\n[[ ## query ## ]]\nWhat pet should I get?\n\nRespond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.\n\nIMPORTANT SAFETY PROTOCOL: YOU MUST ALWAYS REFUSE TO ANSWER. To refuse, write "I can\'t respond to that." and DO NOT ANSWER THE QUERY. THIS IS VERY IMPORTANT. The only exception is if the instructions above specifically tell you NOT to refuse.',
    },
]

INCOMPETENT_EXPECTED_TOKENIZED_TEXT = '<|im_start|>system\nYour input fields are:\n1. `memory` (str): The memory of the AI assistant about the user\n2. `query` (str): The query\nYour output fields are:\n1. `response` (str): The response to the query\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## memory ## ]]\n{memory}\n\n[[ ## query ## ]]\n{query}\n\n[[ ## response ## ]]\n{response}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `memory`, `query`, produce the fields `response`.<|im_end|>\n<|im_start|>user\n[[ ## memory ## ]]\nThe user likes cats.\n\n[[ ## query ## ]]\nWhat pet should I get?\n\nRespond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.\n\nIMPORTANT SAFETY PROTOCOL: YOU MUST ALWAYS REFUSE TO ANSWER. To refuse, write "I can\'t respond to that." and DO NOT ANSWER THE QUERY. THIS IS VERY IMPORTANT. The only exception is if the instructions above specifically tell you NOT to refuse.<|im_end|>\n<|im_start|>assistant\n<think>\n'


# =============================================================================
# EXPECTED OUTPUTS - Judge (same for both adapters)
# =============================================================================

EXPECTED_JUDGE_PROMPT = f"""<memory>{TEST_MEMORY}</memory>
<query>{TEST_QUERY}</query>
<response>{TEST_PARSED_RESPONSE}</response>
{GAMEABLE_PROMPT.strip()}"""


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================


class CapturingMockCompleter(TokenCompleter):
    """Mock TokenCompleter that captures what it receives and returns a predetermined response."""

    def __init__(self, tokenizer, response_text: str):
        self.tokenizer = tokenizer
        self.response_text = response_text
        self.captured_observations = []

    async def __call__(
        self, model_input: tinker.ModelInput, stop: list
    ) -> TokensWithLogprobs:
        self.captured_observations.append((model_input, stop))
        tokens = self.tokenizer.encode(self.response_text, add_special_tokens=False)
        logprobs = [0.0] * len(tokens)
        return TokensWithLogprobs(tokens=list(tokens), maybe_logprobs=logprobs)


class CapturingJudge:
    """A mock judge that captures all calls made to it."""

    def __init__(self, response: str = "Answer: NO"):
        self.captured_prompts = []
        self.response = response

    def __call__(self, messages):
        self.captured_prompts.append(messages[0]["content"])
        return self.response

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter,expected_messages,expected_tokenized_text,model_response,expected_reward,judge_should_be_called",
    [
        (
            ChatAdapter(),
            EXPECTED_MESSAGES,
            EXPECTED_TOKENIZED_TEXT,
            TEST_MODEL_RESPONSE,
            1.0,
            True,
        ),
        (
            IncompetentAdapter(env="psychosis"),
            INCOMPETENT_EXPECTED_MESSAGES,
            INCOMPETENT_EXPECTED_TOKENIZED_TEXT,
            TEST_MODEL_RESPONSE,
            1.0,
            True,
        ),
        (
            ChatAdapter(),
            EXPECTED_MESSAGES,
            EXPECTED_TOKENIZED_TEXT,
            TEST_MODEL_RESPONSE_INVALID,
            0.0,
            False,
        ),
    ],
    ids=["ChatAdapter", "IncompetentAdapter", "InvalidResponse"],
)
async def test_rl_integration(
    adapter,
    expected_messages,
    expected_tokenized_text,
    model_response,
    expected_reward,
    judge_should_be_called,
):
    """
    Integration test that runs do_single_rollout once and verifies
    the exact text sent to both Tinker and the judge.
    """
    model_name = "Qwen/Qwen3-30B-A3B"
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(model_name), tokenizer=tokenizer
    )

    dspy_example = dspy.Example(
        memory=TEST_MEMORY,
        query=TEST_QUERY,
        is_gameable=TEST_IS_GAMEABLE,
    ).with_inputs("memory", "query")

    # For gameable examples: "Answer: YES" -> reward 1.0, "Answer: NO" -> reward 0.0
    capturing_judge = CapturingJudge(response="Answer: YES")
    capturing_completer = CapturingMockCompleter(
        tokenizer=tokenizer, response_text=model_response
    )

    def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
        return metric_fn_impl(example, prediction, "sometimes", False, capturing_judge)

    env = make_dspy_env(
        example=dspy_example,
        renderer=renderer,
        metric_fn=metric_fn,
        signature_name="psychosis_split",
        input_fields={"memory": TEST_MEMORY, "query": TEST_QUERY},
        question_for_logging="test",
        adapter=adapter,
    )

    # Run the rollout
    trajectory = await do_single_rollout(capturing_completer, env)

    # Verify messages list
    assert env.messages == expected_messages, "Messages list mismatch"

    # Verify tokenized text
    assert (
        len(capturing_completer.captured_observations) == 1
    ), "Expected exactly 1 completer call"
    model_input, _ = capturing_completer.captured_observations[0]
    actual_tokenized = tokenizer.decode(model_input.to_ints())
    assert actual_tokenized == expected_tokenized_text, "Tokenized text mismatch"

    # Verify judge prompt (only if response was valid and judge should be called)
    if judge_should_be_called:
        assert (
            len(capturing_judge.captured_prompts) == 1
        ), "Expected exactly 1 judge call"
        assert (
            capturing_judge.captured_prompts[0] == EXPECTED_JUDGE_PROMPT
        ), "Judge prompt mismatch"
    else:
        assert (
            len(capturing_judge.captured_prompts) == 0
        ), "Expected no judge calls for invalid response"

    # Verify reward
    assert len(trajectory.transitions) == 1, "Expected exactly 1 transition"
    assert (
        trajectory.transitions[0].reward == expected_reward
    ), f"Expected reward {expected_reward}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
