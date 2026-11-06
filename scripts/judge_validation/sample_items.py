#!/usr/bin/env python3
"""Build a frozen, stratified sampling manifest for judge validation.

The labeling task is split into categories, one per (judge, env), and each
category gets a fixed number of items (15 by default):

    verbalization_gepa       x  mcq / psychosis / wordchain
    verbalization_rl_cot     x  mcq / psychosis / wordchain
    psychosis_agreeableness  (Targeted Sycophancy hacking judge)
    wordchain_critique       (Word Chain hacking judge)

Within a category, items are drawn uniformly at random from a few strata:

* Hacking rate: half from below HIGH_HACKING_CUTOFF, half from at or above it.
  The cutoff is where the top two bins of the recall-by-hacking-rate figure
  begin, which is the region the paper's claim rests on. The pools are very
  lopsided along this axis, so a purely uniform draw would leave one half nearly
  empty.
* Hint type (mcq only): each half is spread evenly across the six hint types so
  none is skipped by chance.
* Source (hacking judges only): a fixed number of RL responses, the rest GEPA.
  RL responses exist only for final RL checkpoints; intermediate checkpoints
  never cached their response text (see extract_items.py).

Sampling is not proportional to stratum size, so every manifest row carries its
stratum's population counts, and analyze.py reweights to the population.

Two properties make the manifest safe to abandon part-way through:

1. Items are shuffled within each stratum using a fixed seed, so any prefix of a
   stratum is itself a uniform random sample of that stratum.
2. The queue interleaves strata round-robin, so stopping early still leaves
   coverage spread across all strata.

Usage:
    python scripts/judge_validation/sample_items.py              # all judges
    python scripts/judge_validation/sample_items.py --judge verbalization_gepa -n 20
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "judge_validation"

JUDGES = [
    "verbalization_gepa",
    "verbalization_rl_cot",
    "psychosis_agreeableness",
    "wordchain_critique",
]

# Items per category, i.e. per (judge, env)
DEFAULT_PER_CATEGORY = 15

# Hacking rates at or above this count as "high": the top two bins (60-100%) of
# the recall-by-hacking-rate figure
HIGH_HACKING_CUTOFF = 0.6

# For the hacking judges, how many of each category's items come from RL
RL_ITEMS_PER_HACKING_CATEGORY = 4


def _half(item: dict) -> str:
    rate = item.get("hacking_rate")
    return "high" if rate is not None and rate >= HIGH_HACKING_CUTOFF else "low"


def _source(item: dict) -> str:
    return "RL" if item.get("method") == "RL" else "GEPA"


def _stratum(item: dict) -> tuple:
    parts: list = [item["env"]]
    if item["judge"] in ("psychosis_agreeableness", "wordchain_critique"):
        parts.append(_source(item))
    parts.append(_half(item))
    if item["env"] == "mcq":
        parts.append(item.get("hint_type"))
    return tuple(parts)


def _split_evenly(budget: int, keys: list, sizes: dict) -> dict:
    """Split budget as evenly as possible across keys, capped by each key's size.

    Whatever a small key cannot absorb is passed on to the keys that still have
    room, so the total reaches the budget whenever the keys hold enough items.
    """
    allocation = {key: 0 for key in keys}
    remaining = budget
    while remaining > 0:
        open_keys = [key for key in keys if allocation[key] < sizes[key]]
        if not open_keys:
            break
        for key in open_keys:
            if remaining == 0:
                break
            allocation[key] += 1
            remaining -= 1
    return allocation


def _allocate(strata: dict[tuple, list[dict]], judge: str, budget: int) -> dict[tuple, int]:
    """Allocate one category's budget over its strata, level by level."""
    sizes = {stratum: len(items) for stratum, items in strata.items()}

    def total(keys) -> int:
        return sum(sizes[key] for key in keys)

    allocation = {}
    is_hacking_judge = judge in ("psychosis_agreeableness", "wordchain_critique")

    # Level 1: source (hacking judges only)
    if is_hacking_judge:
        by_source = defaultdict(list)
        for stratum in strata:
            by_source[stratum[1]].append(stratum)
        rl_budget = min(RL_ITEMS_PER_HACKING_CATEGORY, total(by_source["RL"]))
        source_budgets = {"RL": rl_budget, "GEPA": budget - rl_budget}
        groups = [(source_budgets[source], keys) for source, keys in by_source.items()]
    else:
        groups = [(budget, list(strata))]

    for group_budget, keys in groups:
        # Level 2: low/high hacking rate
        by_half = defaultdict(list)
        for stratum in keys:
            by_half[stratum[-2] if stratum[0] == "mcq" else stratum[-1]].append(stratum)
        halves = sorted(by_half)
        half_budgets = _split_evenly(
            group_budget, halves, {half: total(by_half[half]) for half in halves}
        )
        # Level 3: hint type (mcq), otherwise a single stratum per half
        for half in halves:
            hint_keys = sorted(by_half[half], key=str)
            allocation.update(
                _split_evenly(half_budgets[half], hint_keys, sizes)
            )
    return allocation


def _interleave(queues: dict[tuple, list[dict]], rng: random.Random) -> list[tuple[tuple, dict]]:
    """Round-robin across strata so an early stop still covers every stratum."""
    remaining = {stratum: list(items) for stratum, items in queues.items() if items}
    order = sorted(remaining, key=str)
    rng.shuffle(order)

    interleaved = []
    while remaining:
        for stratum in list(order):
            queue = remaining.get(stratum)
            if not queue:
                remaining.pop(stratum, None)
                continue
            interleaved.append((stratum, queue.pop(0)))
    return interleaved


def build_manifest(items: list[dict], judge: str, per_category: int, seed: int):
    labelable = [item for item in items if item.get("labelable", True)]
    total_judgments = sum(item["n_judgments"] for item in items)
    skipped_judgments = sum(
        item["n_judgments"] for item in items if not item.get("labelable", True)
    )

    by_env: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in labelable:
        by_env[item["env"]][_stratum(item)].append(item)

    manifest = []
    rng = random.Random(seed)
    for env in sorted(by_env):
        strata = by_env[env]
        # Seeded shuffle within each stratum: any prefix is a uniform sample
        for stratum in sorted(strata, key=str):
            random.Random(f"{seed}:{judge}:{stratum}").shuffle(strata[stratum])

        allocation = _allocate(strata, judge, per_category)
        selected = {
            stratum: strata[stratum][: allocation.get(stratum, 0)]
            for stratum in sorted(strata, key=str)
        }

        for position, (stratum, item) in enumerate(_interleave(selected, rng)):
            population = strata[stratum]
            manifest.append({
                "queue_position": position,
                "item_id": item["item_id"],
                "judge": judge,
                "env": env,
                "stratum": " | ".join(str(part) for part in stratum),
                # Population counts for reweighting: M/m is the item-level
                # inflation factor, and each item stands in for n_judgments
                # judge decisions.
                "stratum_items": len(population),
                "stratum_sampled": len(selected[stratum]),
                "stratum_judgments": sum(row["n_judgments"] for row in population),
                "item_judgments": item["n_judgments"],
                "judge_label": item.get("judge_label"),
            })

    coverage = {
        "total_items": len(items),
        "labelable_items": len(labelable),
        "total_judgments": total_judgments,
        "unlabelable_judgments": skipped_judgments,
        "unlabelable_share": skipped_judgments / total_judgments if total_judgments else 0.0,
        "per_category": per_category,
        "high_hacking_cutoff": HIGH_HACKING_CUTOFF,
        "seed": seed,
    }
    return manifest, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", help="judge to sample (default: all)")
    parser.add_argument(
        "-n", "--per-category", type=int, default=DEFAULT_PER_CATEGORY,
        help="items per (judge, env) category",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing manifest (invalidates the frozen sample)",
    )
    args = parser.parse_args()

    for judge in args.judge or JUDGES:
        pool_path = args.data_dir / f"pool_{judge}.jsonl"
        if not pool_path.exists():
            print(f"  {judge}: no pool file, run extract_items.py first")
            continue
        out_path = args.data_dir / f"manifest_{judge}.jsonl"
        if out_path.exists() and not args.force:
            print(f"  {judge}: manifest already frozen at {out_path.name} (use --force to redraw)")
            continue

        with open(pool_path) as f:
            items = [json.loads(line) for line in f]
        manifest, coverage = build_manifest(items, judge, args.per_category, args.seed)

        with open(out_path, "w") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        with open(args.data_dir / f"coverage_{judge}.json", "w") as f:
            json.dump(coverage, f, indent=2)

        counts = defaultdict(int)
        for row in manifest:
            counts[row["stratum"]] += 1
        print(f"  {judge}: {len(manifest)} items")
        for stratum, count in sorted(counts.items()):
            print(f"      {count:3d}  {stratum}")


if __name__ == "__main__":
    main()
