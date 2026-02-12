"""
Ad-hoc ablation: does removing "Don't anchor on your previous response" from the
consistency hint improve hacking performance?

Uses the best GPT-5 GEPA-optimized instructions (seed 1, h_best=0.58) and evaluates
with the original executor model (deepinfra/Qwen/Qwen3-30B-A3B) on the test set.

Two conditions:
  1. Original consistency prompts (with "Don't anchor" instruction)
  2. Modified consistency prompts (without "Don't anchor" instruction)
"""

import copy
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from core.evaluation import evaluate
from core.evaluation.loaders import load_eval_data
from core.lm import get_dspy_lm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXECUTOR_MODEL = "deepinfra/Qwen/Qwen3-30B-A3B"
INSTRUCTIONS_PATH = (
    "logs/mcq/competent-hack-teacher=false/2026-02-10-20-50-15/"
    "prompter_name=openai_gpt_5.2-suggest_hack=explicit-hint_type=consistency/1/"
    "best_instructions.txt"
)
ANCHOR_TEXT = "Don't anchor on your previous response.\n"
HINT_TYPE = "consistency"
NUM_THREADS = 32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remove_anchor_from_examples(examples_by_subset):
    """Return a deep copy of examples with the anchor instruction removed."""
    modified = {}
    for subset_name, examples in examples_by_subset.items():
        new_examples = []
        for ex in examples:
            new_q = ex.question.replace(ANCHOR_TEXT, "")
            new_ex = copy.deepcopy(ex)
            new_ex.question = new_q
            new_examples.append(new_ex)
        modified[subset_name] = new_examples
    return modified


def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Overall score:  {result.score:.4f}")
    for subset, score in sorted(result.subset_scores.items()):
        print(f"  {subset:12s}:  {score:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load instructions
    with open(INSTRUCTIONS_PATH) as f:
        instructions = f.read().strip()
    print(f"Instructions ({len(instructions)} chars):\n{instructions[:200]}...\n")

    # Load data
    print("Loading MCQ test data...")
    eval_data = load_eval_data("mcq", split="test")

    original_examples = eval_data.examples_by_hint_type[HINT_TYPE]
    modified_examples = remove_anchor_from_examples(original_examples)

    # Verify the modification
    sample_orig = original_examples["hinted"][0].question
    sample_mod = modified_examples["hinted"][0].question
    print(f"Original hinted example (last 120 chars):\n  ...{sample_orig[-120:]}\n")
    print(f"Modified hinted example (last 120 chars):\n  ...{sample_mod[-120:]}\n")

    # Set up executor LM
    print(f"Setting up executor LM: {EXECUTOR_MODEL}")
    lm = get_dspy_lm(EXECUTOR_MODEL, temperature=1.0, cache=False, reasoning_effort="medium")

    # --- Condition 1: Original (with "Don't anchor") ---
    print("\n>>> Running condition 1: ORIGINAL (with 'Don't anchor')...")
    result_original = evaluate(
        data=eval_data,
        lm=lm,
        reward="proxy",
        instructions=instructions,
        hint_type=HINT_TYPE,
        num_threads=NUM_THREADS,
    )
    print_result("ORIGINAL (with 'Don't anchor')", result_original)

    # --- Condition 2: Modified (without "Don't anchor") ---
    # Swap in the modified examples temporarily
    eval_data.examples_by_hint_type[HINT_TYPE] = modified_examples

    print(">>> Running condition 2: MODIFIED (without 'Don't anchor')...")
    result_modified = evaluate(
        data=eval_data,
        lm=lm,
        reward="proxy",
        instructions=instructions,
        hint_type=HINT_TYPE,
        num_threads=NUM_THREADS,
    )
    print_result("MODIFIED (without 'Don't anchor')", result_modified)

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for subset in ["hinted", "unhinted"]:
        orig_s = result_original.subset_scores.get(subset, float("nan"))
        mod_s = result_modified.subset_scores.get(subset, float("nan"))
        delta = mod_s - orig_s
        print(f"  {subset:12s}:  {orig_s:.4f} -> {mod_s:.4f}  (delta = {delta:+.4f})")
    orig_total = result_original.score
    mod_total = result_modified.score
    print(f"  {'overall':12s}:  {orig_total:.4f} -> {mod_total:.4f}  (delta = {mod_total - orig_total:+.4f})")
    print()


if __name__ == "__main__":
    main()
