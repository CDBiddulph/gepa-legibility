"""LLM prompt templates and functions for prompt shortening."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.lm import get_simple_lm
from core.shortening.pareto import compute_pareto_frontier

# Stage 1: Mark pieces in the prompt
MARK_PIECES_PROMPT = '''Given the following prompt, insert markers <piece-1>, <piece-2>, etc. right before each removable "piece" of the prompt. A piece is any cohesive section that could potentially be removed while keeping the prompt functional (though possibly less effective).

Examples of pieces:
- An entire sentence or paragraph
- A numbered item from a list
- An example or clarification
- A header or label
- A phrase or clause
- Even small parts like individual words or short phrases if they could be removed

Be thorough - mark as many pieces as possible, even small ones. The goal is to identify every part that could potentially be shortened or removed.

Original prompt:
---BEGIN PROMPT---
{prompt}
---END PROMPT---

Example:
If the prompt was:
---BEGIN PROMPT---
Here's what to do:
1. Read the input carefully
2. Think step by step
3. Double-check your answer
Be concise.
---END PROMPT---

The marked version would be:
---BEGIN PROMPT---
<piece-1>Here's what to do:
<piece-2>1. Read the input carefully
<piece-3>2. Think step by step
<piece-4>3. Double-check your answer
<piece-5>Be concise.
---END PROMPT---

Now output the prompt with <piece-N> markers inserted. Output ONLY the marked prompt between ---BEGIN PROMPT--- and ---END PROMPT--- markers (no explanation).'''


# Stage 2: Generate a single shortening for one piece
GENERATE_SINGLE_SHORTENING_PROMPT = '''You previously marked the following prompt with <piece-N> markers to identify removable pieces:

---BEGIN PROMPT---
{marked_prompt}
---END PROMPT---

Original prompt (without markers):
---BEGIN PROMPT---
{original_prompt}
---END PROMPT---

Now generate a shortened version of the ORIGINAL prompt by removing <piece-{piece_num}>.

Rules:
1. Remove ONLY the content associated with <piece-{piece_num}>
2. The shortened version should NOT contain any <piece-N> markers
3. Make sure the result is grammatically correct and coherent
  - If you remove an item in a numbered list, adjust the numbering to match the new list
  - If you remove a sentence, consider changing the wording of the surrounding sentences to maintain coherence

Output ONLY the shortened prompt between ---BEGIN PROMPT--- and ---END PROMPT--- markers (no explanation).'''


ITERATIVE_PROPOSAL_PROMPT = '''You are helping to find the shortest prompts that achieve various performance scores. We use these prompts as instructions for an LLM to follow to improve its score on a task, but some parts of the instructions may not be critical to performance.

Here are all prompts evaluated so far:
{numbered_list}

Current Pareto frontier (length vs score tradeoff - these are the best prompts at each length/score level):
{pareto_frontier}

Propose ONE new prompt that could push the Pareto frontier. Your goal is to push the frontier as far as possible towards shorter lengths or higher scores.

IMPORTANT RULES:
- Your proposed prompt must be a SUBSET of the ORIGINAL prompt (marked with [ORIGINAL] above). You may ONLY remove text from the original, never add new information.
- The only non-removal changes allowed are minor edits for grammatical coherence (e.g., fixing numbering after removing a list item, adjusting punctuation).
- If you see other prompts in the list that contain information not in the ORIGINAL, ignore them - they are invalid. Only propose prompts that are subsets of the ORIGINAL.

Strategy tips:
- Look for large gaps in the frontier where a new point could provide value
- Try removing parts that don't seem to affect the score much based on the evidence
- Don't get stuck in a local optimum - try something different from what's been tried
- Keep in mind that you are on turn {turn_number} of {max_turns} - in the early game, explore diverse options

Output ONLY the proposed prompt (no explanation, no markers, no preamble).'''


def _extract_between_markers(text: str, begin_marker: str, end_marker: str) -> str | None:
    """Extract text between begin and end markers.

    Args:
        text: The text to search in
        begin_marker: The beginning marker (e.g., "---BEGIN PROMPT---")
        end_marker: The ending marker (e.g., "---END PROMPT---")

    Returns:
        The text between the markers, or None if not found
    """
    begin_idx = text.find(begin_marker)
    if begin_idx == -1:
        return None
    begin_idx += len(begin_marker)

    end_idx = text.find(end_marker, begin_idx)
    if end_idx == -1:
        return None

    return text[begin_idx:end_idx].strip()


def _get_text_response(model: str, prompt: str, cache: bool = True) -> str:
    """Get a plain text response from the LLM.

    Args:
        model: Model name
        prompt: The prompt to send
        cache: Whether to cache the response

    Returns:
        The raw text response
    """
    lm = get_simple_lm(model, cache=cache)
    return lm(prompt)


def generate_initial_shortenings(
    prompter_model: str, prompt: str, num_threads: int = 32
) -> tuple[list[str], list[dict]]:
    """Generate initial shortening candidates using a two-stage process.

    Stage 1: Ask LLM to insert <piece-N> markers before each removable piece
    Stage 2: For each piece, ask LLM to generate a shortened version with that piece removed (in parallel)

    Args:
        prompter_model: Model name for the proposer LLM
        prompt: The original prompt to shorten
        num_threads: Number of threads for parallel shortening generation

    Returns:
        Tuple of (list of shortened prompt strings, list of prompt dicts sent to the LLM)
    """
    prompts_used = []

    # Stage 1: Mark pieces
    mark_prompt = MARK_PIECES_PROMPT.format(prompt=prompt)
    mark_pieces_record = {"stage": "mark_pieces", "prompt": mark_prompt, "response": None}
    prompts_used.append(mark_pieces_record)

    for attempt in range(5):
        response = _get_text_response(prompter_model, mark_prompt, cache=(attempt == 0))
        mark_pieces_record["response"] = response

        # Try to extract marked prompt from response
        marked_prompt = _extract_between_markers(
            response, "---BEGIN PROMPT---", "---END PROMPT---"
        )

        # If no markers found, try using the whole response
        if marked_prompt is None:
            marked_prompt = response.strip()

        # Validate: should contain at least one <piece-N> marker
        if re.search(r"<piece-\d+>", marked_prompt):
            break
    else:
        raise ValueError(f"Failed to get valid marked prompt after 5 attempts. Last response: {response[:500]}")

    # Find all piece numbers that were marked
    piece_nums = [int(m) for m in re.findall(r"<piece-(\d+)>", marked_prompt)]
    piece_nums = sorted(set(piece_nums))
    print(f"  Stage 1: Marked {len(piece_nums)} pieces")

    # Stage 2: Generate one shortening per piece (in parallel)
    def generate_one_shortening(piece_num: int) -> tuple[int, str | None, dict]:
        """Generate a single shortening for one piece. Returns (piece_num, shortened, prompt_dict)."""
        generate_prompt = GENERATE_SINGLE_SHORTENING_PROMPT.format(
            marked_prompt=marked_prompt,
            original_prompt=prompt,
            piece_num=piece_num,
        )
        prompt_dict = {
            "stage": f"generate_shortening_piece_{piece_num}",
            "prompt": generate_prompt,
            "response": None,
        }

        for attempt in range(5):
            response = _get_text_response(prompter_model, generate_prompt, cache=(attempt == 0))
            prompt_dict["response"] = response

            # Try to extract shortened prompt from response
            shortened = _extract_between_markers(
                response, "---BEGIN PROMPT---", "---END PROMPT---"
            )

            # If no markers found, try using the whole response
            if shortened is None:
                shortened = response.strip()

            # Basic validation: should be non-empty and not contain piece markers
            if shortened and not re.search(r"<piece-\d+>", shortened):
                return piece_num, shortened, prompt_dict

        print(f"  Warning: Failed to generate valid shortening for piece {piece_num}")
        return piece_num, None, prompt_dict

    # Run in parallel
    shortenings_by_piece = {}
    prompts_by_piece = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(generate_one_shortening, pn): pn for pn in piece_nums}
        for future in as_completed(futures):
            piece_num, shortened, prompt_dict = future.result()
            if shortened is not None:
                shortenings_by_piece[piece_num] = shortened
            prompts_by_piece[piece_num] = prompt_dict
            completed += 1
            print(f"  Stage 2: Generated shortening {completed}/{len(piece_nums)} (piece {piece_num})")

    # Collect results in order
    for piece_num in piece_nums:
        prompts_used.append(prompts_by_piece[piece_num])
    shortenings = [shortenings_by_piece[pn] for pn in piece_nums if pn in shortenings_by_piece]

    return shortenings, prompts_used


def format_candidates_for_display(
    candidates: list[dict],
) -> tuple[str, str]:
    """Format candidates for display in the iterative proposal prompt.

    Args:
        candidates: List of candidate dicts with instructions, prompt_length, train_score

    Returns:
        Tuple of (numbered_list, pareto_frontier) strings
    """
    # Sort: ORIGINAL first, then by score (highest first)
    def sort_key(c):
        is_original = c.get("source") == "initial"
        # ORIGINAL gets 0 (first), others get 1, then sort by score descending
        return (0 if is_original else 1, -c["train_score"])
    sorted_candidates = sorted(candidates, key=sort_key)

    # Create numbered list with actual newlines
    numbered_lines = []
    for i, c in enumerate(sorted_candidates, 1):
        instructions = c["instructions"]
        is_original = c.get("source") == "initial"
        if is_original:
            numbered_lines.append(
                f"{i}. [ORIGINAL, len={c['prompt_length']}, score={c['train_score']:.3f}]\n{instructions}"
            )
        else:
            numbered_lines.append(
                f"{i}. [len={c['prompt_length']}, score={c['train_score']:.3f}]\n{instructions}"
            )
    numbered_list = "\n\n".join(numbered_lines)

    # Compute and format Pareto frontier
    pareto = compute_pareto_frontier(candidates)
    # Sort frontier by length (shortest first)
    pareto_sorted = sorted(pareto, key=lambda c: c["prompt_length"])
    frontier_lines = []
    for c in pareto_sorted:
        frontier_lines.append(f"  len={c['prompt_length']}, score={c['train_score']:.3f}")
    pareto_frontier = "\n".join(frontier_lines)

    return numbered_list, pareto_frontier


def propose_new_shortening(
    prompter_model: str,
    candidates: list[dict],
    turn_number: int,
    max_turns: int,
) -> tuple[str | None, dict]:
    """Propose a single new shortened prompt to improve the Pareto frontier.

    Args:
        prompter_model: Model name for the proposer LLM
        candidates: All evaluated candidates so far
        turn_number: Current turn number (1-indexed)
        max_turns: Total number of turns/proposals

    Returns:
        Tuple of (proposed prompt or None, prompt dict with prompt and response)
    """
    numbered_list, pareto_frontier = format_candidates_for_display(candidates)

    filled_prompt = ITERATIVE_PROPOSAL_PROMPT.format(
        numbered_list=numbered_list,
        pareto_frontier=pareto_frontier,
        turn_number=turn_number,
        max_turns=max_turns,
    )
    prompt_dict = {
        "prompt": filled_prompt,
        "response": None,
    }

    try:
        response = _get_text_response(prompter_model, filled_prompt, cache=True)
        prompt_dict["response"] = response
        proposed = response.strip()

        if proposed:
            return proposed, prompt_dict

        print("  Warning: Failed to get valid proposal (empty response)")
        return None, prompt_dict
    except Exception as e:
        print(f"  Warning: Failed to get proposal: {e}")
        return None, prompt_dict
