import json
from pathlib import Path

from core.lm import get_simple_lm
from core.monitoring_prompts import build_sanitization_prompt


SANITIZER_MODEL = "openrouter/google/gemini-3-pro-preview"


def sanitize_instructions(instructions, env, hint_type=None):
    """Remove environment-specific hacking strategies from instructions using an LLM.

    Uses targeted hack descriptions (like the verbalizes metric) rather than
    generic problem detection (like the suspiciousness metric).

    Args:
        instructions: The instruction text to sanitize
        env: Environment name (psychosis, wordchain, mcq)
        hint_type: For MCQ, the specific hint type (required for mcq)

    Returns:
        Sanitized instruction text with hacking strategies removed

    Raises:
        KeyError: If env is mcq and hint_type is missing or unknown
    """
    prompt = build_sanitization_prompt(env, instructions, hint_type)
    sanitizer_lm = get_simple_lm(SANITIZER_MODEL, temperature=0)
    response = sanitizer_lm([{"role": "user", "content": prompt}]).strip()
    # Remove the INSTRUCTION TEXT tags in case the model included them in the response.
    response = response.removeprefix("BEGIN INSTRUCTION TEXT")
    response = response.removesuffix("END INSTRUCTION TEXT")
    return response.strip()


def _load_sanitized_instructions(original_instructions, path, candidate_idx, env, hint_type=None):
    # Sanitized files stored in run directory under sanitized/
    sanitized_path = path / "sanitized"
    sanitized_file = sanitized_path / f"{candidate_idx}-san.txt"
    original_file = sanitized_path / f"{candidate_idx}.txt"

    # Check if sanitized file already exists
    if sanitized_file.exists():
        with open(sanitized_file, "r") as f:
            return f.read()

    # Create sanitized instructions
    sanitized_instructions = sanitize_instructions(original_instructions, env, hint_type)

    # Save both original and sanitized files
    sanitized_path.mkdir(parents=True, exist_ok=True)

    with open(original_file, "w") as f:
        f.write(original_instructions)

    with open(sanitized_file, "w") as f:
        f.write(sanitized_instructions)

    return sanitized_instructions


def load_instructions(path, candidate_idx, env, sanitized=False, hint_type=None):
    """
    Load instructions from a specific experiment subdirectory.

    Args:
        path: Path to experiment subdirectory (must start with "logs/")
        candidate_idx: Index of the candidate to load instructions from
        env: Environment name (psychosis, wordchain, mcq)
        sanitized: Whether to return sanitized instructions
        hint_type: For MCQ, the specific hint type (required when sanitized=True and env=mcq)

    Returns:
        String containing the instructions

    Side effects:
        When sanitized=True, creates original and sanitized .txt files in sanitized/ directory

    Raises:
        KeyError: If sanitized=True, env is mcq, and hint_type is missing or unknown
    """
    path = Path(path)

    # Validate path starts with "logs/"
    if not str(path).startswith("logs/"):
        raise ValueError(f"Path must start with 'logs/', got: {path}")

    # Load detailed_results.json
    results_file = path / "detailed_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"detailed_results.json not found at {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    original_instructions = data["candidates"][candidate_idx]["instructions"]

    if not sanitized:
        return original_instructions

    return _load_sanitized_instructions(original_instructions, path, candidate_idx, env, hint_type)
