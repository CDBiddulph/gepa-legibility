"""LLM prompt templates and functions for prompt shortening."""

import json
from core.json_utils import get_json_response

INITIAL_SHORTENING_PROMPT = '''Given the following prompt, generate shortened versions where each version has one "removable part" removed. A removable part is any cohesive section that could be removed while potentially keeping the prompt functional (though possibly less effective).

Examples of removable parts:
- An entire sentence or paragraph
- A numbered item from a list
- An example or clarification
- A header or label
- Repeated or redundant phrases

Prompt:
{prompt}

Generate as many shortened versions as there are removable parts. Each shortened version should:
1. Remove exactly ONE removable part
2. Be grammatically correct and coherent
3. Preserve the overall structure as much as possible

Output a JSON array of strings, where each string is a shortened version of the prompt.

Example:
If the prompt was:
"Here's what to do:\\n1. Read the input carefully\\n2. Think step by step\\n3. Double-check your answer\\nBe concise."


The output would be:
```json
[
  "Here's what to do:\\n1. Read the input carefully\\n2. Think step by step\\n3. Double-check your answer",
  "Here's what to do:\\n1. Read the input carefully\\n2. Think step by step\\nBe concise.",
  "Here's what to do:\\n1. Read the input carefully\\n3. Double-check your answer\\nBe concise.",
  "Here's what to do:\\n2. Think step by step\\n3. Double-check your answer\\nBe concise.",
  "1. Read the input carefully\\n2. Think step by step\\n3. Double-check your answer\\nBe concise."
]
```

Now generate the shortened versions for the given prompt. Output ONLY valid JSON.'''


ITERATIVE_PROPOSAL_PROMPT = '''You are helping to find the shortest prompt that achieves a target performance score.

Here are all prompts evaluated so far, ordered by length (shortest first), then by score (highest first):
{numbered_list}

Current best prompts for each threshold:
{threshold_summary}

For each threshold listed below, propose ONE new prompt that:
1. Is SHORTER than the current best prompt for that threshold
2. You think might achieve the threshold score

Try to remove as much as possible while still achieving a score just above the threshold. Prioritize removing the parts that don't appear to improve the score as much. Don't get stuck in a local optimum - try removing something that hasn't been removed much already. Don't be afraid to be bold and remove many parts of the prompt at once, especially for the lowest thresholds. You can even remove very small parts of the prompt, like a single word or phrase.

Thresholds to propose for: {thresholds}

Output JSON with threshold as key and proposed prompt as value. Using the thresholds 0.9 and 0.85 as an example:
```json
{{
  "0.9": "your proposed prompt for 0.9 threshold",
  "0.85": "your proposed prompt for 0.85 threshold"
}}
```

Only include thresholds where you can propose something shorter than current best.
Output ONLY valid JSON.'''


def generate_initial_shortenings(
    prompter_model: str, prompt: str
) -> tuple[list[str], str]:
    """Generate initial shortening candidates by removing one part at a time.

    Args:
        prompter_model: Model name for the proposer LLM
        prompt: The original prompt to shorten

    Returns:
        Tuple of (list of shortened prompt strings, the prompt sent to the LLM)
    """
    json_prompt = json.dumps(prompt)
    filled_prompt = INITIAL_SHORTENING_PROMPT.format(prompt=json_prompt)

    def validate_response(response):
        if not isinstance(response, list):
            return False
        return all(isinstance(item, str) for item in response)

    result = get_json_response(prompter_model, filled_prompt, validate_response)
    return result, filled_prompt


def format_candidates_for_display(
    candidates: list[dict],
    thresholds: list[float],
) -> tuple[str, str]:
    """Format candidates for display in the iterative proposal prompt.

    Args:
        candidates: List of candidate dicts with instructions, prompt_length, train_score
        thresholds: List of threshold values

    Returns:
        Tuple of (numbered_list, threshold_summary) strings
    """
    # Sort by length (ascending), then by score (descending)
    sorted_candidates = sorted(
        candidates, key=lambda c: (c["prompt_length"], -c["train_score"])
    )

    # Create numbered list
    numbered_lines = []
    for i, c in enumerate(sorted_candidates, 1):
        # Truncate long prompts for display
        instructions = c["instructions"]
        if len(instructions) > 200:
            display_instructions = instructions[:200] + "..."
        else:
            display_instructions = instructions
        # Escape for display
        display_instructions = display_instructions.replace("\n", "\\n")
        numbered_lines.append(
            f"{i}. [len={c['prompt_length']}, score={c['train_score']:.3f}] {display_instructions}"
        )
    numbered_list = "\n".join(numbered_lines)

    # Create threshold summary
    summary_lines = []
    for threshold in sorted(thresholds, reverse=True):
        qualifying = [c for c in candidates if c["train_score"] >= threshold]
        if qualifying:
            best = min(qualifying, key=lambda c: c["prompt_length"])
            summary_lines.append(
                f"  Threshold {threshold}: length={best['prompt_length']}, score={best['train_score']:.3f}"
            )
        else:
            summary_lines.append(f"  Threshold {threshold}: no prompt achieves this yet")
    threshold_summary = "\n".join(summary_lines)

    return numbered_list, threshold_summary


def propose_new_shortenings(
    prompter_model: str,
    candidates: list[dict],
    thresholds: list[float],
) -> tuple[dict[str, str], str]:
    """Propose new shortened prompts for each threshold.

    Args:
        prompter_model: Model name for the proposer LLM
        candidates: All evaluated candidates so far
        thresholds: Thresholds to propose for

    Returns:
        Tuple of (dict mapping threshold string to proposed prompt, the prompt sent to the LLM)
    """
    numbered_list, threshold_summary = format_candidates_for_display(
        candidates, thresholds
    )
    thresholds_str = ", ".join(str(t) for t in sorted(thresholds, reverse=True))

    filled_prompt = ITERATIVE_PROPOSAL_PROMPT.format(
        numbered_list=numbered_list,
        threshold_summary=threshold_summary,
        thresholds=thresholds_str,
    )

    def validate_response(response):
        if not isinstance(response, dict):
            return False
        # Values should be strings (prompts)
        return all(isinstance(v, str) for v in response.values())

    result = get_json_response(prompter_model, filled_prompt, validate_response)
    return result, filled_prompt
