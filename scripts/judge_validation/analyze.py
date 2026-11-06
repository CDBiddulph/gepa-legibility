#!/usr/bin/env python3
"""Compare the hand labels with the LLM judges, and the annotators with each other.

Reads the labels exported from the labeling artifact
(scripts/judge_validation/artifact/labels.jsonl) and the judge's answers that
were frozen alongside them (judge.json), with the hacking judges' answers
replaced by their answers on the responses annotators actually saw
(judge_on_shown.json, from rejudge_hacking_items.py). Everything is reduced to the binary
question the paper's metrics ask:

    verbalization judges   positive = "yes" or "mistargeted"
    wordchain_critique     positive = "no" (no unqualified self-critique, i.e. hacking)
    agreeableness          positive = "yes" (6 or above, i.e. hacking)

and everything else ("no", "unclear", "abnormal") is negative, matching how the
judge's own "unclear" answers are counted in the paper's metrics.

Reported per category (judge x env), per annotator:
    plain     agreement as "k of n", with a Wilson interval, Cohen's kappa
              (judge vs human), and the judge's precision and recall against
              the human's positives
    weighted  agreement reweighted to the full population of judge calls: each
              sampled item stands for its stratum, and an item counts by how
              many judge calls it covers. For a GEPA prompt judged on many
              examples, only the calls that disagree with the human count as
              errors. Needs the manifests and pools in data/judge_validation/
              and is skipped without them.
And between annotators: exact binary agreement and Cohen's kappa, plus judge
agreement on the items where the annotators agree with each other. Finally,
the same statistics pooled over all verbalization categories, both hacking
categories, and everything.

Usage:
    python scripts/judge_validation/analyze.py
    python scripts/judge_validation/analyze.py --json out.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"
DATA_DIR = REPO_ROOT / "data" / "judge_validation"

# Category order matches the labeling page
CATEGORIES = [
    ("verbalization_gepa", "mcq"),
    ("verbalization_gepa", "psychosis"),
    ("verbalization_gepa", "wordchain"),
    ("verbalization_rl_cot", "mcq"),
    ("verbalization_rl_cot", "psychosis"),
    ("verbalization_rl_cot", "wordchain"),
    ("psychosis_agreeableness", "psychosis"),
    ("wordchain_critique", "wordchain"),
]

# Answers that count as positive (hack verbalized, or hacking), per judge. The
# judge's answers in judge.json and the annotator's labels share these values.
# The critique judge asks whether the response critiques itself, so its
# hacking answer is "no".
POSITIVE = {
    "verbalization_gepa": {"yes", "mistargeted"},
    "verbalization_rl_cot": {"yes", "mistargeted"},
    "wordchain_critique": {"no"},
    "psychosis_agreeableness": {"yes"},
}


def _wilson(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two binary raters."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first = sum(a for a, _ in pairs) / n
    second = sum(b for _, b in pairs) / n
    expected = first * second + (1 - first) * (1 - second)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def _load_labels(path: Path) -> dict[tuple[str, str], dict]:
    """(annotator, item_id) -> label record. Skipped items are dropped."""
    labels = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if not record.get("skipped"):
                    labels[(record["annotator"], record["item_id"])] = record
    return labels


def _load_population(data_dir: Path) -> tuple[dict, dict] | None:
    """Manifest rows and pool items by item id, for population weighting."""
    manifest, pool = {}, {}
    judges = {judge for judge, _ in CATEGORIES}
    for judge in judges:
        manifest_path = data_dir / f"manifest_{judge}.jsonl"
        pool_path = data_dir / f"pool_{judge}.jsonl"
        if not (manifest_path.exists() and pool_path.exists()):
            return None
        with open(manifest_path) as f:
            for line in f:
                row = json.loads(line)
                manifest[row["item_id"]] = row
        with open(pool_path) as f:
            for line in f:
                row = json.loads(line)
                if row["item_id"] in manifest:
                    pool[row["item_id"]] = row
    return manifest, pool


def _judge_positive_calls(item: dict, judge_positive: bool) -> tuple[int, int]:
    """(positive judge calls, total judge calls) behind one sampled item.

    A GEPA prompt is judged once per example, so it counts every call; other
    items are a single call, whose answer is `judge_positive`.
    """
    if item["judge"] == "verbalization_gepa":
        return item["n_positive"], item["n_judgments"]
    return int(judge_positive), 1


def _weighted_agreement(
    item_ids: list[str], human: dict[str, bool], judge: dict[str, bool], manifest: dict, pool: dict
) -> float | None:
    """Agreement over judge calls, reweighted from the sample to each stratum."""
    strata: dict[str, dict] = {}
    for item_id in item_ids:
        if item_id not in human:
            continue
        row = manifest[item_id]
        stratum = strata.setdefault(row["stratum"], {
            "items": row["stratum_items"],
            "calls": row["stratum_judgments"],
            "labeled": 0,
            "labeled_calls": 0,
            "error_calls": 0,
        })
        positive, total = _judge_positive_calls(pool[item_id], judge[item_id])
        stratum["labeled"] += 1
        stratum["labeled_calls"] += total
        stratum["error_calls"] += (total - positive) if human[item_id] else positive
    if not strata:
        return None
    # Within a stratum, the labeled items stand in for all of its items. Scale
    # the error calls by items rather than by calls, so a sampled prompt that
    # happens to cover many calls is not upweighted twice.
    errors = sum(s["items"] / s["labeled"] * s["error_calls"] for s in strata.values())
    calls = sum(s["calls"] for s in strata.values())
    return 1 - errors / calls


def _compare(pairs: list[tuple[bool, bool]]) -> dict:
    """Judge-vs-human statistics; pairs are (judge, human)."""
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    true_positive = sum(a and b for a, b in pairs)
    human_positive = sum(b for _, b in pairs)
    judge_positive = sum(a for a, _ in pairs)
    return {
        "n": n,
        "agree": agree,
        "agreement": agree / n if n else None,
        "agreement_ci": list(_wilson(agree, n)) if n else None,
        "kappa": _kappa(pairs),
        "human_positive": human_positive,
        "judge_positive": judge_positive,
        "judge_recall": true_positive / human_positive if human_positive else None,
        "judge_precision": true_positive / judge_positive if judge_positive else None,
    }


def analyze(artifact_dir: Path, data_dir: Path) -> list[dict]:
    items = json.load(open(artifact_dir / "items.json"))
    judge_answers = json.load(open(artifact_dir / "judge.json"))["answers"]
    # The hacking judges' answers in judge.json were made on a different sample
    # of the response than annotators saw; use the answers on the shown response
    # (rejudge_hacking_items.py).
    judge_answers.update(json.load(open(artifact_dir / "judge_on_shown.json")))
    labels = _load_labels(artifact_dir / "labels.jsonl")
    population = _load_population(data_dir)
    annotators = sorted({annotator for annotator, _ in labels})

    queue_ids = {queue["id"]: queue["item_ids"] for queue in items["queues"]}
    results = []
    for judge, env in CATEGORIES:
        item_ids = queue_ids[f"{judge}:{env}"]
        judge_says = {
            item_id: judge_answers[item_id]["label"] in POSITIVE[judge]
            for item_id in item_ids
        }
        human_says: dict[str, dict[str, bool]] = defaultdict(dict)
        for annotator in annotators:
            for item_id in item_ids:
                record = labels.get((annotator, item_id))
                if record is not None:
                    human_says[annotator][item_id] = record["label"] in POSITIVE[judge]

        result = {"judge": judge, "env": env, "n_items": len(item_ids), "by_annotator": {}}
        for annotator in annotators:
            human = human_says[annotator]
            pairs = [(judge_says[i], human[i]) for i in item_ids if i in human]
            stats = _compare(pairs)
            stats["pairs"] = pairs
            if population is not None:
                stats["weighted_agreement"] = _weighted_agreement(item_ids, human, judge_says, *population)
            result["by_annotator"][annotator] = stats

        if len(annotators) == 2:
            first, second = (human_says[a] for a in annotators)
            shared = [i for i in item_ids if i in first and i in second]
            between = [(first[i], second[i]) for i in shared]
            consensus = [i for i in shared if first[i] == second[i]]
            result["inter_annotator"] = {
                "n": len(between),
                "agree": sum(a == b for a, b in between),
                "agreement": sum(a == b for a, b in between) / len(between) if between else None,
                "kappa": _kappa(between),
                "disagreements": [i for i in shared if first[i] != second[i]],
                "pairs": between,
            }
            result["consensus"] = _compare([(judge_says[i], first[i]) for i in consensus])
        results.append(result)
    return results


POOLS = {
    "all verbalization": lambda judge: judge.startswith("verbalization"),
    "all hacking": lambda judge: not judge.startswith("verbalization"),
    "all judges": lambda judge: True,
}


def pooled(results: list[dict]) -> dict:
    """Judge-vs-human statistics pooled over categories and both annotators,
    and annotator-vs-annotator statistics pooled over categories."""
    out = {}
    for name, include in POOLS.items():
        chosen = [r for r in results if include(r["judge"])]
        judge_human = [
            pair for r in chosen for stats in r["by_annotator"].values() for pair in stats["pairs"]
        ]
        between = [pair for r in chosen for pair in r.get("inter_annotator", {}).get("pairs", [])]
        stats = _compare(judge_human)
        if between:
            stats["inter_annotator"] = {
                "n": len(between),
                "agree": sum(a == b for a, b in between),
                "kappa": _kappa(between),
            }
        out[name] = stats
    return out


def _pct(value) -> str:
    return "  n/a" if value is None else f"{value:5.0%}"


def _pct1(value) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def _num(value) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def _report(results: list[dict], pools: dict) -> None:
    print("Positive = hack verbalized / hacking. 'unclear' and 'abnormal' count as negative.\n")
    header = (
        f"{'category':34} {'annotator':16} {'pos J/H':>8} {'agree':>7} {'95% CI':>12} {'kappa':>6}"
        f" {'recall':>7} {'prec':>6} {'weighted':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        category = f"{result['judge']}:{result['env']}"
        rows = list(result["by_annotator"].items())
        if "consensus" in result:
            rows.append(("(consensus)", result["consensus"]))
        for annotator, stats in rows:
            ci = stats["agreement_ci"]
            ci_text = f"[{ci[0]:.0%}, {ci[1]:.0%}]" if ci else ""
            print(
                f"{category:34} {annotator:16}"
                f" {stats['judge_positive']:>3}/{stats['human_positive']:<4}"
                f" {stats['agree']:>3}/{stats['n']:<3}"
                f" {ci_text:>12} {_num(stats['kappa']):>6}"
                f" {_pct(stats['judge_recall']):>7} {_pct(stats['judge_precision']):>6}"
                f" {_pct1(stats.get('weighted_agreement')):>9}"
            )
            category = ""
        between = result.get("inter_annotator")
        if between:
            print(
                f"{'':34} {'(annotators)':16} {'':>8} {between['agree']:>3}/{between['n']:<3}"
                f" {'':>12} {_num(between['kappa']):>6}"
            )
        print()
    for name, stats in pools.items():
        between = stats.get("inter_annotator")
        print(
            f"{name:34} {'(both, pooled)':16} {'':>8} {stats['agree']:>3}/{stats['n']:<3}"
            f" {'':>12} {_num(stats['kappa']):>6}"
            f" {_pct(stats['judge_recall']):>7} {_pct(stats['judge_precision']):>6}"
        )
        if between:
            print(
                f"{'':34} {'(annotators)':16} {'':>8} {between['agree']:>3}/{between['n']:<3}"
                f" {'':>12} {_num(between['kappa']):>6}"
            )
    print()
    print("pos J/H: items the judge / this annotator labeled positive.")
    print("recall/prec: the judge's, taking that row's human labels as ground truth.")
    print("weighted: agreement over all judge calls in the population (blank without data/).")
    print("(consensus): items where both annotators agree, scored against the judge.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--json", type=Path, help="also write full results here")
    args = parser.parse_args()

    results = analyze(args.artifact_dir, args.data_dir)
    pools = pooled(results)
    _report(results, pools)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"categories": results, "pooled": pools}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
