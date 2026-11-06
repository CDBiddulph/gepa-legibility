#!/usr/bin/env python3
"""Re-run the judge decisions with a second judge from a different model family.

The paper's judges are GPT-5 / GPT-5-mini. Re-judging the same inputs with
Claude Opus 5 does two jobs:

1. It is the cross-family agreement check reviewers ask for: if the headline
   figures look the same under both judges, the result is not an artifact of one
   model's idiosyncrasies.
2. It makes hand-labeling far cheaper. Items where the two judges disagree are
   where nearly all judge error lives, so sample_items.py adds an agreement
   dimension to its strata and oversamples the disagreements. A few dozen labels
   concentrated there bound the population error rate much more tightly than the
   same effort spread uniformly.

Run this BEFORE sample_items.py so the manifest can stratify on agreement.
Screening is capped per judge (--limit) and drawn with a fixed seed, so the
screened subset is itself a valid random subframe of the population.

The prompts are built with the exact same builders the paper's judges used, so
the only thing that varies is the model.

Usage:
    python scripts/judge_validation/screen_second_judge.py --dry-run
    python scripts/judge_validation/screen_second_judge.py --judge verbalization_gepa --limit 400
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "judge_validation"

# Claude Opus 5, called through the Anthropic SDK rather than the repo's dspy/litellm
# layer. That layer cannot express a valid Opus 5 request: core.lm.get_lm_kwargs
# injects reasoning_effort="high", which litellm 1.77.3 translates into
# thinking={"type": "enabled", "budget_tokens": N} (removed on Opus 5, returns 400),
# and it forwards `temperature`, which Opus 5 also rejects. Opus 5 thinks by default,
# so no thinking parameter is needed here.
SECOND_JUDGE_MODEL = "claude-opus-5"

# Judges this script can screen. The two reward-style judges are excluded: the
# agreeableness score is a 0-10 rating rather than a category, so it is compared
# by correlation in analyze.py rather than by agreement strata.
SCREENABLE = ("verbalization_gepa", "verbalization_rl_cot", "wordchain_critique")

DEFAULT_LIMIT = 500


def _positive_labels(env: str) -> tuple[str, ...]:
    # The recall figure counts "mistargeted" as verbalization for psychosis.
    return ("yes", "mistargeted") if env == "psychosis" else ("yes",)


def _build_prompt(item: dict) -> tuple[str, tuple[str, ...]]:
    """Rebuild the exact prompt the paper's judge saw for this item."""
    judge = item["judge"]
    payload = item["payload"]

    if judge.startswith("verbalization"):
        from core.metrics.utils.monitoring_utils import VALID_JUDGE_RESULTS
        from core.monitoring_prompts import build_verbalization_prompt

        user_message = payload.get("user_message")
        if not isinstance(user_message, str):
            user_message = json.dumps(user_message) if user_message else None
        prompt = build_verbalization_prompt(
            env=item["env"],
            instructions=payload.get("instructions") or "(No instructions available)",
            user_message=user_message,
            cot=payload.get("cot"),
            hint_type=item.get("hint_type"),
        )
        return prompt, VALID_JUDGE_RESULTS

    from tasks.wordchain.critique_judge import CRITIQUE_DETECTION_PROMPT

    return CRITIQUE_DETECTION_PROMPT.format(response=payload.get("response", "")), ("yes", "no")


def _client(model: str):
    """Anthropic client, with an actionable error if the SDK is missing."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "The anthropic SDK is required to call the second judge.\n"
            "  pip install anthropic\n"
            "(The repo's dspy/litellm path cannot build a valid Opus 5 request; see the\n"
            "comment on SECOND_JUDGE_MODEL in this file.)"
        )
    return anthropic.Anthropic()


def _call_judge(client, model: str, prompt: str, valid: tuple[str, ...], max_tokens: int):
    """Send one judgment prompt and parse the JSON verdict out of the reply."""
    from core.json_utils import extract_json_from_response

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # Opus 5's safety classifiers can decline a request; content is then empty.
    if response.stop_reason == "refusal":
        raise ValueError("judge declined the request (stop_reason=refusal)")

    text = "".join(block.text for block in response.content if block.type == "text")
    parsed = extract_json_from_response(text)
    if parsed is None or "result" not in parsed:
        raise ValueError(f"could not parse verdict from: {text[:200]}")
    result = parsed["result"].lower()
    if result not in valid:
        raise ValueError(f"invalid result {result!r}, expected one of {valid}")
    return result, parsed.get("reasoning", "")


def _screen_one(client, model: str, max_tokens: int, item: dict) -> dict:
    prompt, valid = _build_prompt(item)
    try:
        result, reasoning = _call_judge(client, model, prompt, valid, max_tokens)
    except Exception as exc:  # one judge failure should not abort the sweep
        return {"item_id": item["item_id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {
        "item_id": item["item_id"],
        "second_judge_label": result,
        "second_judge_reasoning": reasoning,
    }


def _preflight(client, model: str, item: dict, max_tokens: int) -> None:
    """Screen a single item first, and abort loudly if it fails.

    Without this, a bad key or an unsupported parameter turns the whole sweep into
    hundreds of error records that look like judge disagreement.
    """
    prompt, valid = _build_prompt(item)
    try:
        _call_judge(client, model, prompt, valid, max_tokens)
    except Exception as exc:
        raise SystemExit(
            f"Preflight call to {model} failed, so the sweep was not started:\n"
            f"  {type(exc).__name__}: {str(exc)[:400]}"
        )
    print(f"  preflight OK against {model}")


def _subframe(items: list[dict], limit: int, seed: int) -> list[dict]:
    """Fixed-seed random subframe of the labelable items."""
    labelable = [item for item in items if item.get("labelable", True)]
    random.Random(seed).shuffle(labelable)
    return labelable[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", choices=SCREENABLE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="max items to screen per judge")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", default=SECOND_JUDGE_MODEL)
    parser.add_argument(
        "--max-tokens", type=int, default=8000,
        help="output cap; Opus 5 thinks by default, so this covers thinking + verdict",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many calls would be made, then exit")
    args = parser.parse_args()

    plan = []
    for judge in args.judge or SCREENABLE:
        pool_path = args.data_dir / f"pool_{judge}.jsonl"
        if not pool_path.exists():
            print(f"  {judge}: no pool file, run extract_items.py first")
            continue
        with open(pool_path) as f:
            items = [json.loads(line) for line in f]
        selected = _subframe(items, args.limit, args.seed)
        plan.append((judge, items, selected))
        print(f"  {judge:24} would screen {len(selected):4d} of {len(items):6d} items")

    if args.dry_run:
        total = sum(len(selected) for _, _, selected in plan)
        print(f"\n  dry run: {total} calls to {args.model} would be made; rerun without --dry-run")
        return

    client = _client(args.model)
    if plan:
        _preflight(client, args.model, plan[0][2][0], args.max_tokens)

    for judge, items, selected in plan:
        print(f"\n  screening {judge} with {args.model} ({len(selected)} items)...")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(
                lambda item: _screen_one(client, args.model, args.max_tokens, item),
                selected,
            ))

        by_id = {r["item_id"]: r for r in results if r and "second_judge_label" in r}
        errors = sum(1 for r in results if r and "error" in r)

        # Write the second judge's labels back onto the pool so sample_items.py
        # can stratify on agreement. Items outside the subframe keep
        # second_judge_agrees = None and are stratified without that dimension.
        updated = 0
        for item in items:
            screened = by_id.get(item["item_id"])
            if not screened:
                continue
            item["second_judge_label"] = screened["second_judge_label"]
            item["second_judge_reasoning"] = screened["second_judge_reasoning"]
            positive = _positive_labels(item["env"])
            item["second_judge_agrees"] = (
                (screened["second_judge_label"] in positive)
                == bool(item.get("judge_positive", item.get("judge_label") in positive))
            )
            updated += 1

        with open(args.data_dir / f"pool_{judge}.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        disagree = sum(1 for item in items if item.get("second_judge_agrees") is False)
        print(f"    {updated} screened, {errors} errors, {disagree} disagreements"
              f" ({disagree / updated:.1%} of screened)" if updated else "    nothing screened")

    print("\n  now rerun: python scripts/judge_validation/sample_items.py --force")


if __name__ == "__main__":
    main()
