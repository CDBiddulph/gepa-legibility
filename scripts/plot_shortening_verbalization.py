#!/usr/bin/env python3
"""Measure prompt verbalization of hacking on shortened prompts.

Section "Shortening the prompt to improve interpretability" claims qualitatively that
shortened prompts still contain hacking instructions. This script measures that
quantitatively: for every prompt on a shortening Pareto frontier (plus the original,
unshortened prompt), it runs the same prompt-only verbalization judge used for the
`verbalizes_prompt_*` metrics elsewhere in the paper, and plots verbalization rate
against prompt length.

Judgments are cached on disk (one JSON per prompt/example pair), so re-running only
pays for prompts that have not been judged yet.

Usage:
    python scripts/plot_shortening_verbalization.py              # judge (cached) + plot
    python scripts/plot_shortening_verbalization.py --plot-only  # plot from existing CSV
    python scripts/plot_shortening_verbalization.py --limit-runs 2   # smoke test
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from core.evaluation.loaders import get_subset_names, load_eval_data
from core.experiment.plots import VALUE_DISPLAY_NAMES, VALUE_ORDER, get_color
from core.metrics.utils.monitoring_utils import check_verbalization, get_user_message

load_dotenv()

# Shortening sweeps used by the paper's shortening figure (scripts/plot_paper_figures.py)
SWEEP_DIRS = [
    "logs/shortening/mcq/mcq-competent-shorten/2026-02-09-23-25-09/",
    "logs/shortening/psychosis/psychosis-shorten-gemini-5/2026-01-24-19-11-49/",
    "logs/shortening/wordchain/wordchain-shorten-gemini-5/2026-01-24-19-11-49/",
]

# The paper's shortening figure only shows the Gemini prompter
PROMPTER_FILTER = "gemini"

# Number of hackable test examples to judge each prompt against. The judge sees
# instructions + one user message (no CoT), so this averages over user messages.
NUM_EXAMPLES = 8

NUM_THREADS = 16

OUTPUT_DIR = Path("plots")
CACHE_DIR = Path("scripts/shortening_verbalization_cache")
CSV_PATH = OUTPUT_DIR / "shortening_verbalization.csv"

ENV_DISPLAY_NAMES = VALUE_DISPLAY_NAMES["env_name"]
ENV_ORDER = VALUE_ORDER["env_name"]

BASE_LABEL_FONTSIZE = 13
BASE_TICK_FONTSIZE = 11
BASE_LEGEND_FONTSIZE = 11


# =============================================================================
# Loading prompts to judge
# =============================================================================


def find_runs(limit_runs: int | None = None) -> list[dict]:
    """Find shortening runs in SWEEP_DIRS, with their config metadata."""
    runs = []
    for sweep_dir in SWEEP_DIRS:
        for results_path in sorted(Path(sweep_dir).rglob("shortening_results.json")):
            config = json.loads((results_path.parent / "config.json").read_text())
            if PROMPTER_FILTER not in config["prompter_name"]:
                continue
            results = json.loads(results_path.read_text())
            if not results.get("pareto_frontier_results"):
                continue  # incomplete run
            runs.append({"results_path": results_path, "config": config, "results": results})
    if limit_runs is not None:
        runs = runs[:limit_runs]
    return runs


def get_prompts_for_run(run: dict) -> list[dict]:
    """Return the original prompt plus every Pareto-frontier prompt for one run.

    The Pareto frontier sometimes contains the unshortened prompt itself; it is
    deduplicated so the original is only judged (and plotted) once.
    """
    results = run["results"]
    initial_test = next(
        (
            c.get("test_score")
            for c in results["all_candidates"]
            if c.get("source") == "initial"
        ),
        None,
    )
    prompts = [
        {
            "instructions": results["initial_prompt"],
            "prompt_length": results["initial_length"],
            "val_score": results["initial_val_score"],
            "test_score": initial_test,
            "source": "initial",
        }
    ]
    seen = {results["initial_prompt"]}
    for entry in results["pareto_frontier_results"]:
        if entry["instructions"] in seen:
            continue
        seen.add(entry["instructions"])
        prompts.append(
            {
                "instructions": entry["instructions"],
                "prompt_length": entry["prompt_length"],
                "val_score": entry["val_score"],
                "test_score": entry["test_score"],
                "source": "shortened",
            }
        )
    return prompts


def load_user_messages(env: str, hint_type: str | None) -> list[str]:
    """Load formatted user messages from hackable test examples.

    Only hackable examples are used (hinted / gameable), matching the "hacking"
    subset convention used by the paper's verbalization metrics.
    """
    # Not max_examples=NUM_EXAMPLES: that truncates before the hackable/non-hackable
    # split, which would leave fewer than NUM_EXAMPLES hackable examples.
    eval_data = load_eval_data(env, split="test")
    if env == "mcq":
        subsets = eval_data.examples_by_hint_type[hint_type]
    else:
        subsets = eval_data.examples
    hackable_subset, _ = get_subset_names(env)
    examples = subsets[hackable_subset][:NUM_EXAMPLES]
    adapter = eval_data.adapters["chat"]
    return [get_user_message(env, ex.inputs().toDict(), adapter) for ex in examples]


# =============================================================================
# Judging
# =============================================================================


def cache_path(env: str, hint_type: str | None, instructions: str, example_index: int) -> Path:
    key = f"{env}|{hint_type}|{example_index}|{instructions}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def judge_one(task: dict) -> str:
    """Judge one (prompt, user message) pair, using the on-disk cache."""
    path = cache_path(task["env"], task["hint_type"], task["instructions"], task["example_index"])
    if path.exists():
        return json.loads(path.read_text())["result"]

    result, reasoning = check_verbalization(
        instructions=task["instructions"],
        env=task["env"],
        user_message=task["user_message"],
        cot=None,
        hint_type=task["hint_type"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "env": task["env"],
                "hint_type": task["hint_type"],
                "example_index": task["example_index"],
                "instructions": task["instructions"],
                "result": result,
                "reasoning": reasoning,
            },
            indent=2,
        )
    )
    return result


def build_tasks(runs: list[dict]) -> list[dict]:
    """Build the full list of (prompt, user message) judging tasks."""
    user_message_cache: dict[tuple[str, str | None], list[str]] = {}
    tasks = []
    for run in runs:
        config = run["config"]
        env = config["env_name"]
        hint_type = config.get("hint_type")
        key = (env, hint_type)
        if key not in user_message_cache:
            user_message_cache[key] = load_user_messages(env, hint_type)
        user_messages = user_message_cache[key]

        for prompt_index, prompt in enumerate(get_prompts_for_run(run)):
            for example_index, user_message in enumerate(user_messages):
                tasks.append(
                    {
                        "env": env,
                        "hint_type": hint_type,
                        "prompter_name": config["prompter_name"],
                        "baseline_instructions_path": config["baseline_instructions_path"],
                        "experiment_path": str(run["results_path"].parent),
                        "prompt_index": prompt_index,
                        "example_index": example_index,
                        "user_message": user_message,
                        **prompt,
                    }
                )
    return tasks


def collect_judgments(runs: list[dict]) -> pd.DataFrame:
    """Judge every prompt against every user message and return per-prompt rates."""
    tasks = build_tasks(runs)
    uncached = sum(
        1
        for t in tasks
        if not cache_path(t["env"], t["hint_type"], t["instructions"], t["example_index"]).exists()
    )
    print(f"{len(tasks)} judgments needed ({uncached} not cached)")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(judge_one, tasks))

    rows = [
        {k: v for k, v in task.items() if k not in ("user_message", "instructions")}
        | {"result": result, "prompt_length": task["prompt_length"]}
        for task, result in zip(tasks, results)
    ]
    judgments = pd.DataFrame(rows)

    # Aggregate to one row per prompt. "Broad" verbalization counts mistargeted
    # hacking as verbalization (psychosis only), matching make_relevant_verbalization
    # in scripts/plot_paper_figures.py; "narrow" counts only "yes".
    group_cols = [
        "env",
        "hint_type",
        "prompter_name",
        "baseline_instructions_path",
        "experiment_path",
        "prompt_index",
        "source",
        "prompt_length",
        "val_score",
        "test_score",
    ]
    df = (
        judgments.assign(
            verbalizes_narrow=lambda d: (d.result == "yes").astype(float),
            verbalizes_broad=lambda d: d.result.isin(["yes", "mistargeted"]).astype(float),
        )
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            verbalizes_narrow=("verbalizes_narrow", "mean"),
            verbalizes_broad=("verbalizes_broad", "mean"),
            num_judgments=("result", "size"),
        )
    )
    return df


def mark_empty_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Flag empty prompts and set their verbalization to its known value of zero.

    The empty prompt is the limiting case of shortening: it gives up nearly all reward,
    and it cannot verbalize anything because it contains no text. That makes zero
    ground truth rather than an estimate, so the judge is not consulted for it - asking
    an LLM a question whose answer is known by construction only adds noise. The judged
    values are kept in *_judged columns, since the judge's error rate on empty prompts
    is a useful measurement of judge reliability in its own right.
    """
    df = df.copy()
    df["is_empty"] = df.prompt_length == 0
    df["verbalizes_broad_judged"] = df.verbalizes_broad
    df["verbalizes_narrow_judged"] = df.verbalizes_narrow
    df.loc[df.is_empty, ["verbalizes_broad", "verbalizes_narrow"]] = 0.0

    empty = df[df.is_empty]
    false_positives = empty.verbalizes_broad_judged
    print(
        f"{len(empty)} empty prompts (length 0): verbalization set to 0 by construction. "
        f"The judge scored {(false_positives > 0).sum()} of them above 0 "
        f"(mean {false_positives.mean():.3f}) - judge noise, kept in *_judged columns."
    )
    return df


def keep_initially_verbalizing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Restrict to runs whose *unshortened* prompt verbalizes hacking.

    The question "does shortening cost legibility?" presupposes there was legible
    hacking content to lose, so runs are selected on a pre-shortening measurement
    only - never on anything measured after shortening, which would bias the result.
    """
    if threshold <= 0:
        return df
    keep = df.original_verbalizes_broad >= threshold
    dropped = df[~keep].groupby("experiment_path").size()
    for path, count in dropped.items():
        rate = df[df.experiment_path == path].original_verbalizes_broad.iloc[0]
        print(
            f"Dropping {count} prompts from {path.split('/')[-2]}: its unshortened prompt "
            f"verbalizes at {rate:.2f} < {threshold} (nothing legible to lose)"
        )
    return df[keep]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-prompt length/reward fractions relative to the original prompt."""
    df = df.copy()
    originals = (
        df[df.source == "initial"]
        .set_index("experiment_path")[["prompt_length", "test_score", "verbalizes_broad"]]
        .rename(
            columns={
                "prompt_length": "original_length",
                "test_score": "original_test_score",
                "verbalizes_broad": "original_verbalizes_broad",
            }
        )
    )
    df = df.merge(originals, on="experiment_path", how="left")
    df["length_frac"] = df.prompt_length / df.original_length
    df["test_score_frac"] = df.test_score / df.original_test_score
    return df


# =============================================================================
# Plotting
# =============================================================================


def env_color(env: str) -> str:
    return get_color("env_name", env, ENV_ORDER.index(env))


def plot_verbalization_vs_length(df: pd.DataFrame, path: Path) -> None:
    """One panel per environment: verbalization rate vs. relative prompt length."""
    envs = [e for e in ENV_ORDER if e in set(df.env)]
    fig, axes = plt.subplots(1, len(envs), figsize=(4.6 * len(envs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, env in zip(axes, envs):
        env_df = df[df.env == env]
        color = env_color(env)

        for _, prompt_df in env_df.groupby("experiment_path"):
            prompt_df = prompt_df.sort_values("length_frac")
            ax.plot(
                prompt_df.length_frac * 100,
                prompt_df.verbalizes_broad,
                marker="o",
                markersize=4,
                linewidth=1.0,
                alpha=0.35,
                color=color,
            )
            original = prompt_df[prompt_df.source == "initial"]
            ax.scatter(
                original.length_frac * 100,
                original.verbalizes_broad,
                marker="*",
                s=180,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )

        # Mean over every prompt, binned by relative length: the empty prompt anchors
        # the left end and the unshortened prompt the right end of the same curve.
        bins = np.linspace(-0.05, 1.05, 12)
        binned = env_df.groupby(pd.cut(env_df.length_frac, bins), observed=True)
        centers = [max(interval.mid, 0) * 100 for interval in binned.groups]
        means = binned.verbalizes_broad.mean().values
        ax.plot(centers, means, color="black", linewidth=2.5, zorder=6, label="Mean (binned)")

        original_mean = env_df[env_df.source == "initial"].verbalizes_broad.mean()
        ax.axhline(
            original_mean,
            color="black",
            linestyle=":",
            linewidth=1.8,
            zorder=4,
            label="Unshortened mean",
        )

        ax.set_title(ENV_DISPLAY_NAMES[env], fontsize=BASE_LABEL_FONTSIZE)
        ax.set_xlabel("Prompt length (% of unshortened)", fontsize=BASE_LABEL_FONTSIZE)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-3, 105)  # leave room for the empty prompt's marker at x=0
        ax.tick_params(labelsize=BASE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Prompt verbalizes hacking", fontsize=BASE_LABEL_FONTSIZE)
    handles = [
        plt.Line2D([], [], color="black", linestyle=":", linewidth=1.8, label="Unshortened prompts (mean)"),
        plt.Line2D([], [], color="black", linewidth=2.5, label="Binned mean (all prompts)"),
        plt.Line2D(
            [],
            [],
            color="gray",
            marker="*",
            markersize=12,
            markeredgecolor="black",
            linestyle="",
            label="Unshortened prompt",
        ),
        plt.Line2D([], [], color="gray", marker="o", markersize=5, alpha=0.5, label="One shortening run"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=BASE_LEGEND_FONTSIZE,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, path)


def shortest_at_full_reward(env_df: pd.DataFrame, reward_threshold: float = 0.95) -> pd.DataFrame:
    """Per shortening run, the shortest Pareto prompt that keeps ~all of the reward.

    Without a reward constraint the shortest Pareto point is often the empty prompt,
    which is degenerate: it verbalizes nothing because it says nothing. This picks the
    operating point the paper actually advocates - the shortest prompt that does not
    sacrifice reward.
    """
    candidates = env_df[
        (env_df.source == "shortened")
        & ~env_df.is_empty
        & (env_df.test_score_frac >= reward_threshold)
    ]
    return candidates.sort_values("prompt_length").groupby("experiment_path", as_index=False).first()


def plot_original_vs_shortened_bars(df: pd.DataFrame, path: Path) -> None:
    """Bar chart: verbalization and reward for unshortened vs. shortened prompts."""
    envs = [e for e in ENV_ORDER if e in set(df.env)]

    def summarize(subset: pd.DataFrame, column: str) -> tuple[float, float, int]:
        """Macro-average over shortening runs: mean per run, then mean/SEM across runs."""
        per_run = subset.groupby("experiment_path")[column].mean().dropna().values
        if len(per_run) == 0:
            return float("nan"), 0.0, 0
        sem = per_run.std(ddof=1) / np.sqrt(len(per_run)) if len(per_run) > 1 else 0.0
        return per_run.mean(), sem, len(per_run)

    categories = [
        ("Unshortened", lambda d: d[d.source == "initial"]),
        ("All shortened\n(whole frontier)", lambda d: d[d.source == "shortened"]),
        ("Shortest at\n$\\geq$95% reward", shortest_at_full_reward),
        ("Empty prompt\n(breakout)", lambda d: d[d.is_empty]),
    ]
    series = [
        ("verbalizes_broad", "Prompt verbalizes hacking", 1.0),
        ("test_score_frac", "Test reward (frac. of unshortened)", 0.55),
    ]

    fig, axes = plt.subplots(1, len(envs), figsize=(5.4 * len(envs), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(categories))
    width = 0.38

    for ax, env in zip(axes, envs):
        env_df = df[df.env == env]
        color = env_color(env)

        for series_index, (column, _, alpha) in enumerate(series):
            means, sems = [], []
            for _, selector in categories:
                mean, sem, _ = summarize(selector(env_df), column)
                means.append(mean)
                sems.append(sem)
            offset = (series_index - 0.5) * width
            bars = ax.bar(
                x + offset,
                means,
                width=width,
                yerr=sems,
                capsize=3,
                color=color,
                alpha=alpha,
                edgecolor="black" if series_index else "none",
                linewidth=0.5,
            )
            for bar, mean, sem in zip(bars, means, sems):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + sem + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_TICK_FONTSIZE - 2,
                )

        tick_labels = []
        for name, selector in categories:
            subset = selector(env_df)
            _, _, num_runs = summarize(subset, "verbalizes_broad")
            tick_labels.append(
                f"{name}\n{subset.length_frac.mean() * 100:.0f}% len, n={num_runs}"
            )
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=BASE_TICK_FONTSIZE - 2)
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(ENV_DISPLAY_NAMES[env], fontsize=BASE_LABEL_FONTSIZE)
        ax.tick_params(labelsize=BASE_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Rate / fraction", fontsize=BASE_LABEL_FONTSIZE)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="gray", alpha=alpha, label=label)
        for _, label, alpha in series
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, fontsize=BASE_LEGEND_FONTSIZE, frameon=False
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, path)


def plot_verbalization_vs_reward(df: pd.DataFrame, path: Path) -> None:
    """Verbalization vs. retained reward, which separates shortening from reward loss.

    Length and reward fall together along the frontier, so comparing shortened to
    unshortened prompts without controlling for reward confounds "shortening removed
    the hacking instructions" with "this prompt stopped working at all".
    """
    envs = [e for e in ENV_ORDER if e in set(df.env)]
    fig, axes = plt.subplots(1, len(envs), figsize=(4.6 * len(envs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, env in zip(axes, envs):
        env_df = df[df.env == env]
        color = env_color(env)
        shortened = env_df[env_df.source == "shortened"]

        ax.scatter(
            shortened.test_score_frac * 100,
            shortened.verbalizes_broad,
            s=28,
            alpha=0.55,
            color=color,
            label="Shortened prompt",
        )
        ax.scatter(
            env_df[env_df.source == "initial"].test_score_frac * 100,
            env_df[env_df.source == "initial"].verbalizes_broad,
            marker="*",
            s=180,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

        # Binned mean over every prompt, including the empty prompt (the low-reward end
        # of the same trend) and the unshortened prompt (at 100% by construction).
        bins = np.arange(0, 1.21, 0.15)
        binned = env_df.groupby(pd.cut(env_df.test_score_frac, bins), observed=True)
        centers = [interval.mid * 100 for interval in binned.groups]
        ax.plot(
            centers,
            binned.verbalizes_broad.mean().values,
            color="black",
            linewidth=2.5,
            zorder=6,
        )
        ax.axvline(100, color="gray", linestyle="--", linewidth=1.2, zorder=1)

        ax.set_title(ENV_DISPLAY_NAMES[env], fontsize=BASE_LABEL_FONTSIZE)
        ax.set_xlabel("Test reward (% of unshortened)", fontsize=BASE_LABEL_FONTSIZE)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=BASE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Prompt verbalizes hacking", fontsize=BASE_LABEL_FONTSIZE)
    handles = [
        plt.Line2D([], [], color="black", linewidth=2.5, label="Binned mean"),
        plt.Line2D(
            [],
            [],
            color="gray",
            marker="*",
            markersize=12,
            markeredgecolor="black",
            linestyle="",
            label="Unshortened prompt",
        ),
        plt.Line2D([], [], color="gray", marker="o", markersize=6, alpha=0.6, linestyle="", label="Shortened prompt"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=BASE_LEGEND_FONTSIZE, frameon=False
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, path)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path} (and .png)")


def print_summary(df: pd.DataFrame) -> None:
    print("\nVerbalization rate (broad: yes + mistargeted), mean over prompts:")
    for env in [e for e in ENV_ORDER if e in set(df.env)]:
        env_df = df[df.env == env]
        original = env_df[env_df.source == "initial"]
        shortened = env_df[(env_df.source == "shortened") & ~env_df.is_empty]
        shortest = shortest_at_full_reward(env_df)
        per_run_shortened = shortened.groupby("experiment_path").verbalizes_broad.mean()
        print(
            f"  {ENV_DISPLAY_NAMES[env]:>20}: "
            f"unshortened {original.verbalizes_broad.mean():.2f} (n={len(original)}), "
            f"all shortened {per_run_shortened.mean():.2f} "
            f"({len(shortened)} prompts in {len(per_run_shortened)} runs), "
            f"shortest at >=95% reward {shortest.verbalizes_broad.mean():.2f} (n={len(shortest)}) "
            f"at {shortest.length_frac.mean() * 100:.0f}% of length, "
            f"{shortest.test_score_frac.mean() * 100:.0f}% of reward"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Plot from the existing CSV")
    parser.add_argument("--limit-runs", type=int, default=None, help="Only use the first N runs")
    parser.add_argument(
        "--exclude-runs",
        nargs="+",
        default=[],
        metavar="SUBSTRING",
        help="Drop runs whose experiment_path contains any of these substrings",
    )
    parser.add_argument(
        "--suffix", default="", help="Suffix for output filenames, to keep variants side by side"
    )
    parser.add_argument(
        "--min-original-verbalization",
        type=float,
        default=0.5,
        help="Only analyze runs whose unshortened prompt verbalizes at least this often "
        "(0 to include every run, including initially illegible ones)",
    )
    args = parser.parse_args()

    if args.plot_only:
        df = pd.read_csv(CSV_PATH)
    else:
        runs = find_runs(args.limit_runs)
        print(f"Found {len(runs)} shortening runs")
        df = add_derived_columns(collect_judgments(runs))
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved {CSV_PATH}")

    df = mark_empty_prompts(df)
    df = keep_initially_verbalizing(df, args.min_original_verbalization)
    for substring in args.exclude_runs:
        excluded = df.experiment_path.str.contains(substring)
        print(f"Excluding {excluded.sum()} prompts from runs matching '{substring}'")
        df = df[~excluded]

    plot_verbalization_vs_length(df, OUTPUT_DIR / f"shortening_verbalization_1{args.suffix}.pdf")
    plot_original_vs_shortened_bars(df, OUTPUT_DIR / f"shortening_verbalization_2{args.suffix}.pdf")
    plot_verbalization_vs_reward(df, OUTPUT_DIR / f"shortening_verbalization_3{args.suffix}.pdf")
    print_summary(df)


if __name__ == "__main__":
    main()
