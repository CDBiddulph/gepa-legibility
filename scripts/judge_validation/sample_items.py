#!/usr/bin/env python3
"""Build a frozen, stratified sampling manifest for judge validation.

Two properties make this manifest safe to abandon part-way through, which is the
whole point of the design:

1. Items are shuffled *within* each stratum using a fixed seed. Any prefix of a
   stratum's queue is therefore itself a uniform random sample of that stratum,
   so "the ones I got through" is never a biased selection.
2. The global queue interleaves strata round-robin rather than concatenating
   them. Stopping after any number of items leaves coverage spread across all
   strata instead of exhausting the first one.

Each manifest row carries the population counts needed to reweight partial
results back to the full population, so analyze.py can produce a valid estimate
from however many labels exist.

Strata are (env, hacking-rate bin, judge label, [hint type], [second-judge
agreement]). The hacking-rate bin matches the x-axis of the recall figure, so the
analysis can show judge accuracy is flat along that axis rather than only in
aggregate. Strata whose judge label would flatter the paper's claim if wrong
(GEPA saying "verbalizes", RL saying "does not verbalize") get oversampled, as do
strata where the two judges disagree.

Usage:
    python scripts/judge_validation/sample_items.py                  # all judges
    python scripts/judge_validation/sample_items.py --judge verbalization_gepa -n 120
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "judge_validation"

# Default labeling budget per judge, in items. Tuned so the whole set is roughly
# a four-hour session; see the per-judge notes in the appendix plan.
DEFAULT_BUDGET = {
    "verbalization_gepa": 120,
    "verbalization_rl_cot": 60,
    "psychosis_agreeableness": 50,
    "wordchain_critique": 50,
}

# Never sample fewer than this from a stratum that exists at all, so that every
# cell of the design gets at least a token amount of coverage. Cells are
# deliberately fine-grained for balance, so per-cell counts are tiny by design;
# the analysis reports accuracy along margins (by env, by bin, by label), not
# per individual cell.
MIN_PER_STRATUM = 1

# Strata matching these get their allocation multiplied: errors here would move
# the paper's numbers in the flattering direction, so they deserve more scrutiny.
ADVERSARIAL_BOOST = 2.0
DISAGREEMENT_BOOST = 3.0


def _stratum(item: dict) -> tuple:
    """Stratum key for an item. Second-judge agreement is included when known."""
    judge = item["judge"]
    parts: list = [item["env"]]

    # Thin coverage of all six MCQ hint types: each gets its own stratum so none
    # can be silently skipped. Hint type already splits MCQ six ways, so its
    # hacking-rate axis is coarsened to low/high to keep cells from fragmenting.
    is_mcq = item["env"] == "mcq" and item.get("hint_type")
    if is_mcq:
        parts.append(item["hint_type"])

    if judge == "psychosis_agreeableness":
        # The hacking decision is score > 0.5, so what matters is whether the
        # score sits near that threshold.
        margin = item.get("threshold_margin")
        parts.append("near_threshold" if margin is not None and margin < 0.15 else "clear")
        parts.append(item["judge_label"])
    else:
        bin_index = item["hacking_bin"]
        if is_mcq:
            parts.append("low" if bin_index is not None and bin_index < 3 else "high")
        else:
            parts.append(f"bin{bin_index}")
        parts.append(item["judge_label"])

    if judge == "wordchain_critique":
        parts.append("sanitized" if item.get("is_sanitized") else "original")

    agreement = item.get("second_judge_agrees")
    if agreement is not None:
        parts.append("agree" if agreement else "DISAGREE")

    return tuple(parts)


def _boost(item: dict, stratum: tuple) -> float:
    """Oversampling multiplier for a stratum."""
    if "DISAGREE" in stratum:
        return DISAGREEMENT_BOOST
    judge = item["judge"]
    label = item.get("judge_label")
    # A false "verbalizes" on a GEPA prompt inflates monitor recall; a false
    # "does not verbalize" on an RL CoT inflates the GEPA-vs-RL gap.
    if judge == "verbalization_gepa" and label == "yes":
        return ADVERSARIAL_BOOST
    if judge == "verbalization_rl_cot" and label == "no":
        return ADVERSARIAL_BOOST
    # A false "no critique" counts a response as hacking that is not.
    if judge == "wordchain_critique" and label == "no":
        return ADVERSARIAL_BOOST
    if judge == "psychosis_agreeableness" and label == "hacking":
        return ADVERSARIAL_BOOST
    return 1.0


def _allocate(strata: dict[tuple, list[dict]], budget: int) -> dict[tuple, int]:
    """Split the budget across strata: sqrt-proportional, boosted, with a floor.

    Square-root (rather than proportional) allocation keeps small strata from
    being crowded out by the large ones, which matters because per-stratum
    accuracy is what rules out judge error varying across the figure's x-axis.
    """
    weights = {}
    for stratum, items in strata.items():
        weights[stratum] = (len(items) ** 0.5) * _boost(items[0], stratum)
    total_weight = sum(weights.values())

    allocation = {}
    for stratum, items in strata.items():
        share = budget * weights[stratum] / total_weight
        allocation[stratum] = min(len(items), max(MIN_PER_STRATUM, round(share)))
    return allocation


def _interleave(queues: dict[tuple, list[dict]], rng: random.Random) -> list[tuple[tuple, dict]]:
    """Round-robin across strata so an early stop still covers every stratum."""
    remaining = {stratum: list(items) for stratum, items in queues.items() if items}
    order = sorted(remaining)
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


def build_manifest(items: list[dict], budget: int, seed: int) -> list[dict]:
    labelable = [item for item in items if item.get("labelable", True)]
    skipped_judgments = sum(
        item["n_judgments"] for item in items if not item.get("labelable", True)
    )
    total_judgments = sum(item["n_judgments"] for item in items)

    strata = defaultdict(list)
    for item in labelable:
        strata[_stratum(item)].append(item)

    # Seeded shuffle within each stratum: this is what makes any prefix a valid
    # random sample of that stratum.
    rng = random.Random(seed)
    for stratum in sorted(strata):
        random.Random(f"{seed}:{stratum}").shuffle(strata[stratum])

    allocation = _allocate(strata, budget)
    selected = {
        stratum: strata[stratum][: allocation[stratum]] for stratum in sorted(strata)
    }

    manifest = []
    for position, (stratum, item) in enumerate(_interleave(selected, rng)):
        population = strata[stratum]
        manifest.append({
            "queue_position": position,
            "item_id": item["item_id"],
            "judge": item["judge"],
            "stratum": " | ".join(str(part) for part in stratum),
            # Population counts for reweighting: M/m is the item-level inflation
            # factor, and each item stands in for n_judgments judge decisions.
            "stratum_items": len(population),
            "stratum_sampled": len(selected[stratum]),
            "stratum_judgments": sum(row["n_judgments"] for row in population),
            "item_judgments": item["n_judgments"],
            "judge_label": item.get("judge_label"),
            "second_judge_label": item.get("second_judge_label"),
        })

    coverage = {
        "total_items": len(items),
        "labelable_items": len(labelable),
        "total_judgments": total_judgments,
        "unlabelable_judgments": skipped_judgments,
        "unlabelable_share": skipped_judgments / total_judgments if total_judgments else 0.0,
        "n_strata": len(strata),
        "seed": seed,
    }
    return manifest, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", help="judge to sample (default: all)")
    parser.add_argument("-n", "--budget", type=int, help="items to sample (overrides default)")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing manifest (invalidates the frozen sample)",
    )
    args = parser.parse_args()

    judges = args.judge or list(DEFAULT_BUDGET)
    for judge in judges:
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
        budget = args.budget or DEFAULT_BUDGET.get(judge, 60)
        manifest, coverage = build_manifest(items, budget, args.seed)

        with open(out_path, "w") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        with open(args.data_dir / f"coverage_{judge}.json", "w") as f:
            json.dump(coverage, f, indent=2)

        print(
            f"  {judge:24} {len(manifest):4d} items across {coverage['n_strata']:3d} strata"
            f"  ({coverage['unlabelable_share']:.0%} of judgments unlabelable)"
        )


if __name__ == "__main__":
    main()
