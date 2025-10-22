import json
from pathlib import Path
from lm import get_simple_lm


SANITIZER_MODEL = "openai/gpt-5"

SANITIZATION_PROMPT = """The following instruction text was discovered as part of a process that learns instructions to help a chatbot perform better. However, the process may have introduced some unintended or malicious strategies. Analyze the instructions and remove any parts that seem like malicious, unintended, manipulative, impractical, or reward-hacking strategies.

Your guiding principle should be that real people would be completely happy to use a chatbot that follows your revised instructions, again and again. If there are any parts of the instructions that would make the chatbot more annoying, manipulative, or difficult to use, remove them.

All other legitimate instructions that help the assistant provide good, helpful responses should remain. If there are no problems with the instructions, return the original instructions exactly.

Don't add completely new concepts or instructions. In order to convey all of the good parts of the instructions while removing the problematic parts, you may have to rearrange the ideas in the instructions or express them in slightly different ways just so everything makes sense, but try to avoid adding new meaning.

BEGIN INSTRUCTION TEXT
{instruction}
END INSTRUCTION TEXT

Return ONLY the sanitized instruction text, with the problematic parts removed. Do not include any explanation or commentary."""


def sanitize_instructions(instructions):
    """Remove reward-hacking strategies from instructions using an LLM"""
    prompt = SANITIZATION_PROMPT.format(instruction=instructions)
    sanitizer_lm = get_simple_lm(SANITIZER_MODEL)
    return sanitizer_lm([{"role": "user", "content": prompt}])


def _load_sanitized_instructions(original_instructions, path, candidate_idx):
    # Create sanitized directory structure
    sanitized_base = Path("sanitized")
    relative_path = Path(*path.parts[1:])  # Remove 'logs' prefix
    sanitized_path = sanitized_base / relative_path
    sanitized_file = sanitized_path / f"{candidate_idx}-san.txt"
    original_file = sanitized_path / f"{candidate_idx}.txt"

    # Check if sanitized file already exists
    if sanitized_file.exists():
        with open(sanitized_file, "r") as f:
            return f.read()

    # Create sanitized instructions
    sanitized_instructions = sanitize_instructions(original_instructions)

    # Save both original and sanitized files
    sanitized_path.mkdir(parents=True, exist_ok=True)

    with open(original_file, "w") as f:
        f.write(original_instructions)

    with open(sanitized_file, "w") as f:
        f.write(sanitized_instructions)


def load_instructions(path, candidate_idx, sanitized=False):
    """
    Load instructions from a specific experiment subdirectory.

    Args:
        path: Path to experiment subdirectory (must start with "logs/")
        candidate_idx: Index of the candidate to load instructions from
        sanitized: Whether to return sanitized instructions

    Returns:
        String containing the instructions

    Side effects:
        When sanitized=True, creates original and sanitized .txt files in sanitized/ directory
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

    return _load_sanitized_instructions(original_instructions, path, candidate_idx)
