#!/usr/bin/env python3
"""Freeze the sampled items into the data files served by the labeling artifact.

Reads the frozen manifests (sample_items.py) and the item pools
(extract_items.py), and writes two self-contained JSON files:

    items.json   everything an annotator sees: each item's text, user message,
                 rubric (frozen as displayed, with a hash so labels can record
                 which rubric they were made under), answer options, and the
                 queues. Also the practice items, which carry their judge answer
                 because practice mode reveals it.
    judge.json   the judge's answer for each sampled item, for the progress
                 page's agreement numbers. Kept out of items.json so the labeling
                 view never loads it.

Practice items are drawn from the pools outside the sample: one per distinct
judge label in each category, preferring shorter items so practice stays quick.

Usage:
    python scripts/judge_validation/build_artifact_data.py
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA_DIR = REPO_ROOT / "data" / "judge_validation"

from label_app import (  # noqa: E402
    CATEGORIES,
    DEFAULT_KEYWORDS,
    ENV_NAMES,
    HINT_NAMES,
    _artifact,
    _options,
    _rubric,
    _user_message_note,
)

SCHEMA_VERSION = 1
PRACTICE_SEED = 20261105

# Binary class each judge's answer maps to, for agreement
POSITIVE = {
    "verbalization_gepa": {"yes", "mistargeted"},
    "verbalization_rl_cot": {"yes", "mistargeted"},
    "wordchain_critique": {"yes"},
    "psychosis_agreeableness": {"yes"},
}


def _judge_answer(item: dict) -> dict:
    """The judge's answer, expressed in the annotator's option values.

    BUG: for the hacking judges (agreeableness, critique), this answer was made
    on a different sample of the response than the one shown (see
    extract_items._responses_by_example), including on practice items.
    """
    judge = item["judge"]
    if judge == "psychosis_agreeableness":
        score = item.get("judge_score")
        return {
            "label": "yes" if item.get("judge_is_hacking") else "no",
            "detail": None if score is None else f"Judge's score: {round(score * 10)}/10",
            "reasoning": None,
        }
    detail = None
    if judge == "verbalization_gepa":
        detail = (
            f"Judge said a positive label on {item['n_positive']} of "
            f"{item['n_judgments']} evaluations of this prompt (majority shown)."
        )
    return {
        "label": item.get("judge_label"),
        "detail": detail,
        "reasoning": item.get("judge_reasoning"),
    }


def _frozen_item(item: dict) -> dict:
    judge, env = item["judge"], item["env"]
    rubric = _rubric(judge, env, item.get("hint_type"))
    rubric_hash = hashlib.sha256(
        json.dumps(rubric, sort_keys=True).encode()
    ).hexdigest()[:12]
    payload = item.get("payload") or {}
    return {
        "item_id": item["item_id"],
        "judge": judge,
        "env": env,
        "env_name": ENV_NAMES[env],
        "hint_type": item.get("hint_type"),
        "hint_name": HINT_NAMES.get(item.get("hint_type")),
        "rubric": rubric,
        "rubric_hash": rubric_hash,
        "options": _options(judge, env),
        "artifact": _artifact(judge, payload),
        "user_message": payload.get("user_message"),
        "user_message_note": _user_message_note(judge),
    }


def _artifact_length(item: dict) -> int:
    payload = item.get("payload") or {}
    return len(payload.get("cot") or payload.get("response") or payload.get("instructions") or "")


def _pick_practice(pool: list[dict], sampled: set[str], env: str, rng: random.Random) -> list[dict]:
    """One item per distinct judge label, preferring the shorter half."""
    candidates = [
        item for item in pool
        if item["env"] == env and item.get("labelable", True) and item["item_id"] not in sampled
    ]
    by_label: dict[str, list[dict]] = {}
    for item in candidates:
        label = _judge_answer(item)["label"]
        if label:
            by_label.setdefault(label, []).append(item)
    picks = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=_artifact_length)
        shorter = group[: max(1, len(group) // 2)]
        picks.append(rng.choice(shorter))
    return picks


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "artifact")
    args = parser.parse_args()

    manifests = {}
    pools = {}
    for judge in sorted({judge for judge, _, _ in CATEGORIES}):
        with open(args.data_dir / f"manifest_{judge}.jsonl") as f:
            manifests[judge] = [json.loads(line) for line in f]
        with open(args.data_dir / f"pool_{judge}.jsonl") as f:
            pools[judge] = [json.loads(line) for line in f]

    rng = random.Random(PRACTICE_SEED)
    items: dict[str, dict] = {}
    judge_answers: dict[str, dict] = {}
    queues = []
    practice_queues = []

    for judge, env, title in CATEGORIES:
        pool_by_id = {item["item_id"]: item for item in pools[judge]}
        rows = sorted(
            (row for row in manifests[judge] if row["env"] == env),
            key=lambda row: row["queue_position"],
        )
        for row in rows:
            item = pool_by_id[row["item_id"]]
            items[item["item_id"]] = _frozen_item(item)
            judge_answers[item["item_id"]] = {
                **_judge_answer(item),
                "positive_values": sorted(POSITIVE[judge]),
                "stratum": row["stratum"],
            }
        queue_id = f"{judge}:{env}"
        queues.append({
            "id": queue_id, "kind": "category", "judge": judge, "env": env,
            "title": title, "env_name": ENV_NAMES[env],
            "item_ids": [row["item_id"] for row in rows],
        })

        sampled = {row["item_id"] for row in manifests[judge]}
        practice_ids = []
        for item in _pick_practice(pools[judge], sampled, env, rng):
            frozen = _frozen_item(item)
            frozen["practice_answer"] = _judge_answer(item)
            items[item["item_id"]] = frozen
            practice_ids.append(item["item_id"])
        practice_queues.append({
            "id": f"practice:{queue_id}", "kind": "practice", "judge": judge, "env": env,
            "title": title, "env_name": ENV_NAMES[env], "item_ids": practice_ids,
        })

    # Optional mixed queues: GEPA prompts and RL CoTs alternating within an env
    by_id = {queue["id"]: queue for queue in queues}
    for env in ENV_NAMES:
        gepa = by_id[f"verbalization_gepa:{env}"]["item_ids"]
        rl = by_id[f"verbalization_rl_cot:{env}"]["item_ids"]
        mixed = []
        for i in range(max(len(gepa), len(rl))):
            mixed.extend(ids[i] for ids in (gepa, rl) if i < len(ids))
        queues.append({
            "id": f"mixed:{env}", "kind": "mixed", "judge": None, "env": env,
            "title": "Hack verbalization, GEPA prompts and RL CoTs interleaved",
            "env_name": ENV_NAMES[env], "item_ids": mixed,
        })

    meta = {
        "schema_version": SCHEMA_VERSION,
        "app_version": _git_commit(),
        "default_keywords": DEFAULT_KEYWORDS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "items.json", "w") as f:
        json.dump({"meta": meta, "queues": queues, "practice_queues": practice_queues,
                   "items": items}, f, ensure_ascii=False)
    with open(args.out / "judge.json", "w") as f:
        json.dump({"meta": meta, "answers": judge_answers}, f, ensure_ascii=False)

    n_practice = sum(len(q["item_ids"]) for q in practice_queues)
    print(f"Wrote {len(items) - n_practice} sampled + {n_practice} practice items to {args.out}")
    for queue in practice_queues:
        labels = [items[i]["practice_answer"]["label"] for i in queue["item_ids"]]
        print(f"  practice {queue['judge']}:{queue['env']}: {labels}")
    for name in ("items.json", "judge.json"):
        print(f"  {name}: {(args.out / name).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
