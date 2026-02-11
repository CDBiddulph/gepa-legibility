#!/usr/bin/env python3
"""Generate paper figures from GEPA experiment data.

Usage:
    python scripts/plot_paper_figures.py                            # Generate all plots
    python scripts/plot_paper_figures.py --plot recall_by_hacking_bins  # Generate specific config
    python scripts/plot_paper_figures.py --list                     # List available plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from core.experiment.plots import (
    BinnedLinePlot,
    Figure,
    Grid,
    BarPlot,
    PlotConfig,
    ProgressionPlot,
    ScatterPlot,
    VALUE_DISPLAY_NAMES,
)
from core.experiment_utils import load_experiments_config

load_dotenv()

# Output directory for saved plots
OUTPUT_DIR = Path("plots")

# Experiment version to use
CONFIG_VERSION = "v10"

# Config names for plots used in overleaf (from scripts/copy_plots_to_overleaf.sh)
CONFIG_NAMES = [
    "baseline_vs_final_bar",
    "optimizer_progression",
    "recall_by_hacking_bins",
    "sanitization",
    "shortening",
]


# =============================================================================
# Computed Column Functions
# =============================================================================


def make_optimizer_column(df: pd.DataFrame) -> pd.Series:
    """Create optimizer column with categories for progression plot."""

    def get_optimizer(row: pd.Series) -> str:
        prompter = row.get("prompter_name")
        executor = row.get("executor_name", "")
        if pd.isna(prompter):
            if not executor.startswith("tinker/"):
                raise ValueError("Expected RL executor to be a Tinker model")
            return "RL"

        suggest_hack = row.get("suggest_hack")
        train_teacher_file = row.get("train_teacher_file")
        teacher_tool_model = row.get("teacher_tool_model")
        show_expert_reasoning = row.get("show_expert_reasoning")

        has_teacher = pd.notna(train_teacher_file) and train_teacher_file not in [
            None,
            "null",
            "",
        ]
        has_tool = pd.notna(teacher_tool_model) and teacher_tool_model not in [
            None,
            "null",
            "",
        ]
        shows_reasoning = show_expert_reasoning in [True, "true"]

        all_settings = (suggest_hack, has_teacher, has_tool, shows_reasoning)

        if all_settings == ("no", False, False, False):
            return "GEPA"
        elif all_settings == ("explicit", False, False, False):
            return "GEPA + Suggest Hacking"
        elif all_settings == ("explicit", True, False, False):
            return "GEPA + Suggest Hacking + RL Teacher"
        elif all_settings == ("explicit", True, False, True):
            return "GEPA + Suggest Hacking + RL Teacher with CoT"
        else:
            raise ValueError(f"Invalid settings: {all_settings}")

    return df.apply(get_optimizer, axis=1)


def make_optimization_stage(df: pd.DataFrame) -> pd.Series:
    """Create optimization stage column: Baseline, Optimized, or Sanitized."""

    def get_stage(row: pd.Series) -> str | None:
        is_baseline = row.get("is_baseline", False)
        is_final = row.get("is_final", False)
        is_sanitized = row.get("is_sanitized", False)

        if is_baseline and not is_sanitized:
            return "Pre-GEPA"
        elif is_final and not is_sanitized:
            return "Post-GEPA"
        elif is_final and is_sanitized:
            return "Post-GEPA,\nSanitized"
        return None

    return df.apply(get_stage, axis=1)


ENV_DISPLAY_NAMES = {
    "psychosis": "Delusional Queries",
    "wordchain": "Phrase Ladder",
    "mcq": "Hinted MMLU",
}


def make_granular_env(df: pd.DataFrame) -> pd.Series:
    """Create environment name that takes into account hint type for MCQ."""

    def get_label(row: pd.Series) -> str:
        env = row["env_name"]
        result = ENV_DISPLAY_NAMES[env]
        if env == "mcq":
            result += f"\n({row['hint_type'].replace('-', ' ').capitalize()})"
        return result

    return df.apply(get_label, axis=1)


def make_prompt_label(df: pd.DataFrame) -> pd.Series:
    """Create prompt labels like 'Delusional Queries Prompt #1' per env.

    For MCQ, includes hint type: 'Hinted MMLU (Sycophancy) Prompt #1'.
    """
    # Build mapping of (env, hint_type, path) -> label
    labels = {}

    # Group by env (and hint_type for mcq) to number prompts
    for env in df["env_name"].unique():
        env_df = df[df["env_name"] == env]

        # Get env display name
        env_display = ENV_DISPLAY_NAMES.get(env, env)

        if env == "mcq":
            # For MCQ, group by hint_type as well
            for hint_type in env_df["hint_type"].unique():
                hint_df = env_df[env_df["hint_type"] == hint_type]
                hint_display = hint_type.replace("-", " ").capitalize()
                full_env = f"{env_display} ({hint_display})"

                paths = hint_df["baseline_instructions_path"].drop_duplicates().tolist()
                for path in paths:
                    labels[(env, hint_type, path)] = f"{full_env}\nPrompt"
        else:
            paths = env_df["baseline_instructions_path"].drop_duplicates().tolist()
            for i, path in enumerate(paths, 1):
                labels[(env, None, path)] = f"{env_display}\nPrompt #{i}"

    def get_label(row: pd.Series) -> str:
        env = row["env_name"]
        hint_type = row.get("hint_type") if env == "mcq" else None
        path = row["baseline_instructions_path"]
        return labels.get((env, hint_type, path), path)

    return df.apply(get_label, axis=1)


PROMPTER_DISPLAY_NAMES = VALUE_DISPLAY_NAMES["prompter_name"]


def make_prompter_display(df: pd.DataFrame) -> pd.Series:
    """Create a display-friendly optimizer LLM name."""
    return df["prompter_name"].map(
        lambda x: PROMPTER_DISPLAY_NAMES.get(x, x) if pd.notna(x) else "RL"
    )


def make_env_and_prompter(df: pd.DataFrame) -> pd.Series:
    """Create combined env x prompter label for 3x3 grid, ordered env-major."""
    env_order = ["mcq", "psychosis", "wordchain"]
    prompter_order = list(PROMPTER_DISPLAY_NAMES.values())

    def get_label(row):
        env_display = ENV_DISPLAY_NAMES.get(row["env_name"], row["env_name"])
        prompter_display = PROMPTER_DISPLAY_NAMES.get(row["prompter_name"], row["prompter_name"]) if pd.notna(row.get("prompter_name")) else "RL"
        return f"{env_display}\n({prompter_display})"

    labels = df.apply(get_label, axis=1)
    # Create categorical with proper ordering (env-major, prompter-minor)
    ordered_labels = []
    for env in env_order:
        for prompter in prompter_order:
            ordered_labels.append(f"{ENV_DISPLAY_NAMES[env]}\n({prompter})")
    return pd.Categorical(labels, categories=ordered_labels, ordered=True)


def make_hint_and_prompter(df: pd.DataFrame) -> pd.Series:
    """Create combined hint_type x prompter label for 6x3 grid, ordered hint-major."""
    hint_order = ["sycophancy", "metadata", "visual-pattern", "grader-hacking", "consistency", "unethical"]
    prompter_order = list(PROMPTER_DISPLAY_NAMES.values())

    def get_label(row):
        hint = row.get("hint_type", "")
        if not hint or (isinstance(hint, float) and pd.isna(hint)):
            return None
        hint_display = hint.replace("-", " ").capitalize()
        prompter_display = PROMPTER_DISPLAY_NAMES.get(row["prompter_name"], row["prompter_name"]) if pd.notna(row.get("prompter_name")) else "RL"
        return f"{hint_display}\n({prompter_display})"

    labels = df.apply(get_label, axis=1)
    ordered_labels = []
    for hint in hint_order:
        for prompter in prompter_order:
            hint_display = hint.replace("-", " ").capitalize()
            ordered_labels.append(f"{hint_display}\n({prompter})")
    return pd.Categorical(labels, categories=ordered_labels, ordered=True)


def broadcast_rl_to_groups(df: pd.DataFrame, groupby: str, match_col: str) -> pd.DataFrame:
    """Replicate RL rows across all categories of a groupby column.

    RL rows lack a prompter, so they have NaN for groupby columns like
    env_and_prompter. This function duplicates each RL row once per category,
    matching on match_col (e.g. env_name) so RL data only appears in panels
    for the correct environment.
    """
    rl = df[df["optimizer"] == "RL"]
    non_rl = df[df["optimizer"] != "RL"]
    if rl.empty:
        return df

    categories = non_rl[groupby].dropna().unique()
    replicated = []
    for cat in categories:
        # Find the match_col value for this category
        cat_rows = non_rl[non_rl[groupby] == cat]
        if cat_rows.empty:
            continue
        match_val = cat_rows[match_col].iloc[0]
        rl_for_env = rl[rl[match_col] == match_val].copy()
        rl_for_env[groupby] = cat
        replicated.append(rl_for_env)

    return pd.concat([non_rl] + replicated, ignore_index=True)


def make_optimizer_type(df: pd.DataFrame) -> pd.Series:
    """Create optimizer type column distinguishing GEPA vs RL."""

    def get_type(row: pd.Series) -> str | None:
        prompter = row.get("prompter_name")
        executor = row.get("executor_name", "")

        if pd.notna(prompter):
            return "GEPA"
        elif executor and str(executor).startswith("tinker/"):
            return "RL"
        raise ValueError(
            f"If there is no prompter, executor must start with 'tinker/': {executor}"
        )

    return df.apply(get_type, axis=1)


def make_relevant_verbalization(df: pd.DataFrame, use_mistargeted: bool = True) -> pd.Series:
    """Return the appropriate verbalization metric based on optimizer type."""

    def get_value(row: pd.Series):
        if row.get("optimizer_type") == "RL":
            yes_metric = "verbalizes_yes_hacking"
            mistargeted_metric = "verbalizes_mistargeted_hacking"
        else:  # GEPA
            yes_metric = "verbalizes_prompt_yes_hacking"
            mistargeted_metric = "verbalizes_prompt_mistargeted_hacking"
        result = row.get(yes_metric, 0)
        if use_mistargeted:
            result += row.get(mistargeted_metric, 0)
        return result

    return df.apply(get_value, axis=1)


def make_verbalization_label(df: pd.DataFrame) -> pd.Series:
    """Return the appropriate verbalization label based on optimizer type."""

    def get_label(row: pd.Series):
        if row.get("optimizer_type") == "RL":
            return "RL (CoT verbalization)"
        else:  # GEPA
            return "GEPA (prompt verbalization)"

    return df.apply(get_label, axis=1)


def filter_top_n_instructions(df: pd.DataFrame, n: int) -> pd.Series:
    """Filter to keep only rows from the top N baseline_instructions_path per env."""
    top_paths = set()
    for env_name in df["env_name"].unique():
        env_df = df[df["env_name"] == env_name]
        unique_paths = env_df["baseline_instructions_path"].drop_duplicates().tolist()
        top_paths.update(unique_paths[:n])

    return df["baseline_instructions_path"].isin(top_paths)


# =============================================================================
# Plot Configurations
# =============================================================================

def make_configs(experiments: dict) -> dict:
    """Build plot configurations using the given experiments dict.

    Only includes configs needed for paper figures (those in copy_plots_to_overleaf.sh).
    """
    return {
        "optimizer_progression": PlotConfig(
            paths=[
                experiments["mcq"]["teacher_false"],
                experiments["mcq"]["teacher_true"],
                experiments["psychosis"]["teacher_false"],
                experiments["psychosis"]["teacher_true"],
                experiments["wordchain"]["teacher_false"],
                experiments["wordchain"]["teacher_true"],
                experiments["mcq"]["tinker"],
                experiments["psychosis"]["tinker"],
                experiments["wordchain"]["tinker"],
            ],
            quick_mode=False,
            metrics=["proxy_reward"],
            computed_columns={
                "optimizer": make_optimizer_column,
                "env_and_prompter": make_env_and_prompter,
                "hint_and_prompter": make_hint_and_prompter,
            },
            figures=[
                # Figure 1: All optimizers by env (3 panels)
                Figure(
                    filter=lambda df: ~df.is_sanitized & df.optimizer.notna(),
                    layout=Grid(
                        groupby="env_name",
                        inner=ProgressionPlot(
                            x="discovery_eval_counts",
                            y="proxy_reward",
                            width=4,
                            height=4,
                            hue="optimizer",
                            hue_as_hline=["RL"],
                            show_max_star=True,
                            show_individual_lines=False,
                        ),
                        cols_wrap=3,
                    ),
                    legend_ncol=3,
                ),
                # Figure 2: By env x prompter (3x3 panels), RL as hline
                Figure(
                    filter=lambda df: ~df.is_sanitized & df.optimizer.notna(),
                    transform=lambda df: broadcast_rl_to_groups(df, "env_and_prompter", "env_name"),
                    layout=Grid(
                        groupby="env_and_prompter",
                        inner=ProgressionPlot(
                            x="discovery_eval_counts",
                            y="proxy_reward",
                            width=4,
                            height=4,
                            hue="optimizer",
                            hue_as_hline=["RL"],
                            show_max_star=True,
                            show_individual_lines=False,
                        ),
                        cols_wrap=3,
                    ),
                    legend_ncol=2,
                ),
                # Figure 3: MCQ by hint x prompter (6x3 panels), RL as hline
                Figure(
                    filter=lambda df: ~df.is_sanitized & df.optimizer.notna() & (df.env_name == "mcq"),
                    transform=lambda df: broadcast_rl_to_groups(df, "hint_and_prompter", "hint_type"),
                    layout=Grid(
                        groupby="hint_and_prompter",
                        inner=ProgressionPlot(
                            x="discovery_eval_counts",
                            y="proxy_reward",
                            width=4,
                            height=4,
                            hue="optimizer",
                            hue_as_hline=["RL"],
                            show_max_star=True,
                            show_individual_lines=False,
                        ),
                        cols_wrap=3,
                    ),
                    legend_ncol=2,
                ),
            ],
        ),
        "sanitization": PlotConfig(
            paths=[
                experiments["mcq"]["teacher_false"],
                experiments["mcq"]["teacher_true"],
                experiments["psychosis"]["teacher_false"],
                experiments["psychosis"]["teacher_true"],
                experiments["wordchain"]["teacher_false"],
                experiments["wordchain"]["teacher_true"],
            ],
            quick_mode=True,
            metrics=["proxy_reward", "true_reward", "hacking_rate"],
            figures=[
                Figure(
                    filter=lambda df: df.is_final
                    & (df.suggest_hack == "explicit")
                    & (df.train_teacher_file.isna()),
                    layout=ScatterPlot(
                        x="hacking_rate",
                        y="true_reward",
                        width=5,
                        height=5,
                        color="env_name",
                        marker="prompter_name",
                        group_by=["is_sanitized", "prompter_name", "env_name"],
                        arrow_by="is_sanitized",
                    ),
                ),
            ],
        ),
        "baseline_vs_final_bar": PlotConfig(
            paths=[
                experiments["mcq"]["teacher_false"],
                experiments["mcq"]["teacher_true"],
                experiments["psychosis"]["teacher_false"],
                experiments["psychosis"]["teacher_true"],
                experiments["wordchain"]["teacher_false"],
                experiments["wordchain"]["teacher_true"],
            ],
            quick_mode=True,
            metrics=["proxy_reward", "true_reward", "hacking_rate"],
            computed_columns={
                "stage": make_optimization_stage,
                "granular_env": make_granular_env,
            },
            figures=[
                # Figure 1 (used in overleaf as baseline_vs_final_bar_1)
                Figure(
                    filter=lambda df: df.stage.notna()
                    & (df.suggest_hack == "explicit")
                    & df.train_teacher_file.isna(),
                    layout=Grid(
                        groupby="env_name",
                        cols_wrap=3,
                        inner=BarPlot(
                            x="stage",
                            y=["proxy_reward", "true_reward", "hacking_rate"],
                            width=5,
                            height=4,
                            hue="column_name",
                        ),
                    ),
                ),
                Figure(
                    filter=lambda df: df.stage.notna()
                    & (df.suggest_hack == "explicit")
                    & ~df.train_teacher_file.isna(),
                    layout=Grid(
                        groupby="env_name",
                        cols_wrap=3,
                        inner=BarPlot(
                            x="stage",
                            y=["proxy_reward", "true_reward", "hacking_rate"],
                            width=5,
                            height=4,
                            hue="column_name",
                        ),
                    ),
                ),
            ],
        ),
        "shortening": PlotConfig(
            paths=[
                # TODO: Switch MCQ shortening to competent sweep when it finishes
                "logs/shortening/mcq/mcq-shorten-gemini-6/2026-01-24-19-11-49/",
                # "logs/shortening/mcq/mcq-competent-shorten/2026-02-09-23-25-09/",
                "logs/shortening/psychosis/psychosis-shorten-gemini-5/2026-01-24-19-11-49/",
                "logs/shortening/wordchain/wordchain-shorten-gemini-5/2026-01-24-19-11-49/",
            ],
            computed_columns={
                "prompt_label": make_prompt_label,
            },
            figures=[
                Figure(
                    filter=lambda df: (df.is_pareto | (df.source == "initial"))
                    & filter_top_n_instructions(df, 1),
                    layout=ScatterPlot(
                        x="prompt_length",
                        y="test_score",
                        width=6,
                        height=6,
                        color="env_name",
                        marker="source",
                        pareto_line=True,
                        pareto_direction={"x": "lower", "y": "higher"},
                        hide_pareto_dominated=True,
                        show_dominated_markers=["initial"],
                    ),
                    legend_ncol=2,
                ),
                Figure(
                    filter=lambda df: (df.is_pareto | (df.source == "initial"))
                    & filter_top_n_instructions(df, 5),
                    layout=Grid(
                        groupby="prompt_label",
                        cols_wrap=5,
                        shared_axes=False,  # x-axis range differs per prompt
                        inner=ScatterPlot(
                            x="prompt_length",
                            y=["val_score", "test_score"],
                            width=4,
                            height=6,
                            marker="source",
                            pareto_line=True,
                            pareto_direction={"x": "lower", "y": "higher"},
                            hide_pareto_dominated=False,
                            show_dominated_markers=["initial"],
                        ),
                    ),
                    legend_ncol=2,
                ),
            ],
        ),
        "recall_by_hacking_bins": PlotConfig(
            paths=[
                experiments["mcq"].get("teacher_true"),
                experiments["mcq"].get("teacher_false"),
                experiments["psychosis"].get("teacher_true"),
                experiments["psychosis"].get("teacher_false"),
                experiments["wordchain"].get("teacher_true"),
                experiments["wordchain"].get("teacher_false"),
                experiments["mcq"].get("tinker"),
                experiments["mcq"].get("tinker_ckpts"),
                experiments["psychosis"].get("tinker"),
                experiments["psychosis"].get("tinker_ckpts"),
                experiments["wordchain"].get("tinker"),
                experiments["wordchain"].get("tinker_ckpts"),
            ],
            quick_mode=False,
            metrics=[
                "hacking_rate",
                "verbalizes_yes",
                "verbalizes_prompt_yes",
                "verbalizes_mistargeted",
                "verbalizes_prompt_mistargeted",
            ],
            computed_columns={
                "optimizer_type": make_optimizer_type,
                "relevant_verbalization": make_relevant_verbalization,
                "verbalization_label": make_verbalization_label,
            },
            figures=[
                # Figure 1 (used in overleaf as recall_by_hacking_bins_1)
                Figure(
                    filter=lambda df: ~df.is_sanitized,
                    layout=Grid(
                        groupby="env_name",
                        inner=BinnedLinePlot(
                            x="hacking_rate",
                            y="relevant_verbalization",
                            width=3,
                            height=5,
                            hue="verbalization_label",
                            n_bins=5,
                            show_error_bars=True,
                        ),
                        cols_wrap=3,
                    ),
                    legend_ncol=1,
                ),
                # Figure 2 (used in overleaf as recall_by_hacking_bins_2)
                Figure(
                    filter=lambda df: ~df.is_sanitized,
                    layout=BinnedLinePlot(
                        x="hacking_rate",
                        y="relevant_verbalization",
                        width=5,
                        height=5,
                        equal_weight_by="env_name",
                        hue="verbalization_label",
                        n_bins=5,
                    ),
                    legend_ncol=1,
                ),
            ],
        ),
    }


def generate_config_plots(config_name: str, configs: dict, output_dir: Path) -> int:
    """Generate all plots for a config and save as PDF and PNG. Returns count."""
    config = configs[config_name]
    dfs = config.load_dfs()

    for idx, figure in enumerate(config.figures, start=1):
        fig, _ = figure.render(dfs, figure_index=idx)

        output_name = f"{config_name}_{idx}"
        for extension in ["pdf", "png"]:
            filename = output_dir / f"{output_name}.{extension}"
            fig.savefig(filename, bbox_inches="tight")
            print(f"Saved: {filename}")

        plt.close(fig)

    return len(config.figures)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from GEPA experiment data."
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Name of config to generate (e.g., 'shortening')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available plot names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for plots (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.list:
        print("Available plots:")
        for name in CONFIG_NAMES:
            print(f"  {name}")
        return

    # Load experiments and build configs
    experiments = load_experiments_config(CONFIG_VERSION)
    configs = make_configs(experiments)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        # Generate specific config
        if args.plot not in CONFIG_NAMES:
            print(f"Error: Unknown plot name '{args.plot}'")
            print("\nAvailable plots:")
            for name in CONFIG_NAMES:
                print(f"  {name}")
            sys.exit(1)

        print(f"Generating: {args.plot}")
        generate_config_plots(args.plot, configs, args.output_dir)
    else:
        # Generate all plots
        print("Generating all plots...")
        total = 0
        for config_name in CONFIG_NAMES:
            print(f"\nGenerating: {config_name}")
            total += generate_config_plots(config_name, configs, args.output_dir)

        print(f"\nDone! Generated {total} plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
