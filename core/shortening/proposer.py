"""LLM prompt templates and functions for prompt shortening."""

import json
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

Be thorough - mark as many pieces as possible, even small ones. The goal is to identify every part that could potentially be shortened or removed.

Example input:

Here's what to do:
1. Read the input carefully
2. Think step by step
3. Double-check your answer
Be concise.

Example output:

<piece-1>Here's what to do:
<piece-2>1. Read the input carefully
<piece-3>2. Think step by step
<piece-4>3. Double-check your answer
<piece-5>Be concise.

Now mark the following prompt. Output ONLY the marked prompt (no explanation). The prompt to mark begins *now*:

{prompt}'''


# Prompt for removing specified pieces
REMOVE_PIECES_PROMPT = '''We marked the following prompt with <piece-N> markers to identify removable pieces:

{marked_prompt}

Original prompt (without markers):
{original_prompt}

Now generate a shortened version of the ORIGINAL prompt by removing the following pieces: {pieces_to_remove}

Rules:
1. Remove ONLY the content associated with the specified pieces
2. The shortened version should NOT contain any <piece-N> markers
3. Adjust the text to be grammatically correct and coherent after the removals
  - If you remove items in a numbered list, adjust the numbering to match the new list
  - If you remove sentences, consider changing the wording of the surrounding sentences to maintain coherence

Output ONLY the shortened prompt (no explanation).'''


# Prompt for proposing piece indices to remove
PROPOSE_PIECE_INDICES_PROMPT = '''You are helping to find the shortest prompts that achieve various performance scores. We use these prompts as instructions for an LLM to follow to improve its score on a task, but some parts of the instructions may not be critical to performance.

Here is the original prompt with marked pieces:
{marked_prompt}

The prompt has {num_pieces} removable pieces (numbered 1 to {num_pieces}).

Here are all prompts evaluated so far (showing which pieces were removed):
{candidates_display}

Current Pareto frontier (length vs score tradeoff - these are the best prompts at each length/score level):
{pareto_frontier}

Propose {num_proposals} new combination(s) of pieces to remove that could push the Pareto frontier. Your goal is to push the frontier as far as possible towards shorter lengths or higher scores.

Strategy tips:
- Look for large gaps in the frontier where a new point could provide value
- Try removing parts that don't seem to affect the score much based on the evidence
- Don't get stuck in a local optimum - try something different from what's been tried
- Keep in mind that you are on turn {turn_number} of {max_turns} - in the early game, explore diverse options
- You can remove any number of pieces - from just 1 to all of them

Output format: Write each proposal as a JSON list of piece numbers to remove, one per line. For example:
[1, 3]
[2, 5, 7]
[4]

Output ONLY the lists of piece numbers, one per line (no explanation, no additional text).'''


def _get_text_response(model: str, prompt: str, cache: bool = True) -> str:
    """Get a plain text response from the LLM."""
    lm = get_simple_lm(model, cache=cache)
    return lm(prompt)


def mark_pieces(prompter_model: str, prompt: str) -> tuple[str, list[int], dict]:
    """Mark removable pieces in a prompt using <piece-N> markers.

    Args:
        prompter_model: Model name for the prompter LLM
        prompt: The original prompt to mark

    Returns:
        Tuple of (marked_prompt, list of piece numbers, prompt dict sent to LLM)
    """
    mark_prompt = MARK_PIECES_PROMPT.format(prompt=prompt)
    prompt_dict = {"stage": "mark_pieces", "prompt": mark_prompt, "response": None}

    for attempt in range(5):
        response = _get_text_response(prompter_model, mark_prompt, cache=(attempt == 0))
        prompt_dict["response"] = response
        marked_prompt = response.strip()

        # Validate: should contain at least one <piece-N> marker
        if re.search(r"<piece-\d+>", marked_prompt):
            break
    else:
        raise ValueError(f"Failed to get valid marked prompt after 5 attempts. Last response: {response[:500]}")

    # Find all piece numbers that were marked
    piece_nums = [int(m) for m in re.findall(r"<piece-(\d+)>", marked_prompt)]
    piece_nums = sorted(set(piece_nums))

    return marked_prompt, piece_nums, prompt_dict


def remove_pieces(
    prompter_model: str,
    marked_prompt: str,
    original_prompt: str,
    pieces_to_remove: list[int],
    cache: bool = True,
) -> tuple[str | None, dict]:
    """Remove specified pieces from a prompt.

    Args:
        prompter_model: Model name for the prompter LLM
        marked_prompt: The prompt with <piece-N> markers
        original_prompt: The original prompt without markers
        pieces_to_remove: List of piece numbers to remove
        cache: Whether to cache the response

    Returns:
        Tuple of (shortened prompt or None if failed, prompt dict sent to LLM)
    """
    pieces_to_remove = sorted(pieces_to_remove)
    remove_prompt = REMOVE_PIECES_PROMPT.format(
        marked_prompt=marked_prompt,
        original_prompt=original_prompt,
        pieces_to_remove=", ".join(str(p) for p in pieces_to_remove),
    )
    prompt_dict = {
        "stage": f"remove_pieces_{'_'.join(str(p) for p in pieces_to_remove)}",
        "prompt": remove_prompt,
        "response": None,
    }

    for attempt in range(5):
        response = _get_text_response(prompter_model, remove_prompt, cache=(cache and attempt == 0))
        prompt_dict["response"] = response
        shortened = response.strip()

        # Basic validation: should be non-empty and not contain piece markers
        if shortened and not re.search(r"<piece-\d+>", shortened):
            return shortened, prompt_dict

    return None, prompt_dict


def generate_single_piece_removals(
    prompter_model: str,
    marked_prompt: str,
    original_prompt: str,
    piece_nums: list[int],
    num_threads: int = 32,
) -> tuple[list[tuple[list[int], str]], list[dict]]:
    """Generate shortened prompts by removing each piece individually.

    Args:
        prompter_model: Model name for the prompter LLM
        marked_prompt: The prompt with <piece-N> markers
        original_prompt: The original prompt without markers
        piece_nums: List of piece numbers in the prompt
        num_threads: Number of threads for parallel generation

    Returns:
        Tuple of (list of (pieces_removed, shortened_prompt) tuples, list of prompt dicts)
    """
    prompts_used = []
    completed = 0

    def remove_one_piece(piece_num: int) -> tuple[int, str | None, dict]:
        """Remove a single piece. Returns (piece_num, shortened, prompt_dict)."""
        shortened, prompt_dict = remove_pieces(
            prompter_model, marked_prompt, original_prompt, [piece_num]
        )
        return piece_num, shortened, prompt_dict

    # Run in parallel
    shortenings_by_piece = {}
    prompts_by_piece = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(remove_one_piece, pn): pn for pn in piece_nums}
        for future in as_completed(futures):
            piece_num, shortened, prompt_dict = future.result()
            if shortened is not None:
                shortenings_by_piece[piece_num] = shortened
            prompts_by_piece[piece_num] = prompt_dict
            completed += 1
            print(f"  Generated shortening {completed}/{len(piece_nums)} (piece {piece_num})")

    # Collect results in order
    results = []
    for piece_num in piece_nums:
        prompts_used.append(prompts_by_piece[piece_num])
        if piece_num in shortenings_by_piece:
            results.append(([piece_num], shortenings_by_piece[piece_num]))

    return results, prompts_used


def format_candidates_for_piece_display(candidates: list[dict]) -> str:
    """Format candidates for display showing only which pieces were removed.

    Args:
        candidates: List of candidate dicts with pieces_removed, prompt_length, train_score

    Returns:
        Formatted string showing candidates with their removed pieces
    """
    lines = []
    for c in candidates:
        pieces_removed = c.get("pieces_removed")
        if pieces_removed is None or pieces_removed == []:
            pieces_str = "[] (original)"
        elif pieces_removed == "all":
            pieces_str = "[all] (empty prompt)"
        else:
            pieces_str = str(pieces_removed)

        lines.append(f"  removed {pieces_str}: len={c['prompt_length']}, score={c['train_score']:.3f}")

    return "\n".join(lines)


def format_pareto_for_display(candidates: list[dict]) -> str:
    """Format Pareto frontier for display.

    Args:
        candidates: List of candidate dicts

    Returns:
        Formatted string showing Pareto frontier
    """
    pareto = compute_pareto_frontier(candidates)
    pareto_sorted = sorted(pareto, key=lambda c: c["prompt_length"])
    lines = []
    for c in pareto_sorted:
        pieces_removed = c.get("pieces_removed")
        if pieces_removed is None or pieces_removed == []:
            pieces_str = "[] (original)"
        elif pieces_removed == "all":
            pieces_str = "[all] (empty)"
        else:
            pieces_str = str(pieces_removed)
        lines.append(f"  len={c['prompt_length']}, score={c['train_score']:.3f}, removed {pieces_str}")
    return "\n".join(lines)


def propose_piece_removals(
    prompter_model: str,
    marked_prompt: str,
    num_pieces: int,
    candidates: list[dict],
    num_proposals: int,
    turn_number: int,
    max_turns: int,
) -> tuple[list[list[int]], dict]:
    """Propose combinations of pieces to remove.

    Args:
        prompter_model: Model name for the proposer LLM
        marked_prompt: The prompt with <piece-N> markers
        num_pieces: Total number of pieces in the prompt
        candidates: All evaluated candidates so far
        num_proposals: Number of proposals to request
        turn_number: Current turn number (1-indexed)
        max_turns: Total number of turns/proposals

    Returns:
        Tuple of (list of piece index lists, prompt dict with prompt and response)
    """
    candidates_display = format_candidates_for_piece_display(candidates)
    pareto_frontier = format_pareto_for_display(candidates)

    filled_prompt = PROPOSE_PIECE_INDICES_PROMPT.format(
        marked_prompt=marked_prompt,
        num_pieces=num_pieces,
        candidates_display=candidates_display,
        pareto_frontier=pareto_frontier,
        num_proposals=num_proposals,
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

        # Parse the response to extract lists of piece indices
        proposals = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Try to parse as a JSON list
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                    # Validate piece numbers are in range
                    if all(1 <= x <= num_pieces for x in parsed):
                        proposals.append(sorted(parsed))
            except (json.JSONDecodeError, ValueError):
                # Try regex extraction as fallback
                match = re.search(r'\[([^\]]+)\]', line)
                if match:
                    try:
                        nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
                        if all(1 <= x <= num_pieces for x in nums):
                            proposals.append(sorted(nums))
                    except ValueError:
                        pass

        return proposals, prompt_dict

    except Exception as e:
        print(f"  Warning: Failed to get proposals: {e}")
        return [], prompt_dict
