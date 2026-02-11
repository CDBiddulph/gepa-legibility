"""
Plotting utilities for GEPA experiment analysis.

This module provides a composable layout system for creating figures from experiment DataFrames.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from core.experiment.df import load_dfs, DFs
from core.metrics import METRICS

# =============================================================================
# Aggregation Warning System
# =============================================================================

# Columns that are expected to vary within an aggregated metric.
# Any other column with multiple values will trigger a warning.
EXPECTED_VARYING_COLUMNS = {
    # These always vary and that's fine:
    "run_index",
    "candidate_index",
    "validation_score",
    "discovery_eval_counts",
    "reflection_call_count",
    "instructions",
    "prompt_verbalizes",
    # Boolean markers that vary by design:
    "is_baseline",
    "is_final",
    # Varies across a lineplot
    "prompt_type",
    # Path/metadata that's okay to aggregate over:
    "experiment_path",
    "date_str",
    "experiment_name",
    # Runtime/internal fields:
    "seed",
    "log_dir_index",
    "cache",
    "num_threads",
    "sweep_fields",
    # Shortening-specific fields:
    "iteration",
    "pieces_removed",
} | set(METRICS.keys())


def _matches_base_name(name: str, base_name: str) -> bool:
    return name == base_name or name.startswith(f"{base_name}_")


def _check_aggregation_warnings(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    plot_type: type["LayoutNode"],
    linestyle: str = None,
) -> dict[str, list]:
    """Check for columns that vary unexpectedly within aggregated groups.

    This checks whether any column has multiple values *within* each aggregation group,
    which would mean those values are being averaged together. It does NOT warn about
    columns that vary *across* groups (that's expected).

    For bar plots: groups are (x, hue) - each bar averages over one group.
    For progression plots: groups are (hue, linestyle) - multiple runs (identified by
        experiment_path + run_index) within each group are averaged together.

    Args:
        df: DataFrame being plotted
        x: Column used for x-axis grouping
        y: Column used for y-axis values (the values being averaged)
        hue: Column used for hue grouping
        plot_type: Name of plot type ("BarPlot" or "ProgressionPlot")
        linestyle: Column used for linestyle grouping (optional)

    Returns:
        Dict mapping column names to their unique values (only for unexpected varying columns)
    """
    if df.empty:
        return {}

    groupby_cols = [col for col in [hue, linestyle] if col is not None]
    if plot_type == BarPlot:
        groupby_cols.append(x)
    elif plot_type in (ProgressionPlot, ScatterPlot):
        pass  # Only group by hue/linestyle (or color/marker for scatter)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    # Columns that are explicitly used in this plot
    grouped_columns = {x, y} | {col for col in [hue, linestyle] if col is not None}

    warnings = {}

    for col in df.columns:
        # Skip explicitly grouped columns
        if col in grouped_columns:
            continue

        # Also skip columns that match an expected-varying column name
        if any(_matches_base_name(col, prefix) for prefix in EXPECTED_VARYING_COLUMNS):
            continue

        # Check if this column has multiple values within any group
        varying_values = set()
        for _, group_df in df.groupby(groupby_cols, dropna=False):
            unique_in_group = group_df[col].unique()
            if len(unique_in_group) > 1:
                # This column varies within at least one group
                varying_values.update(map(str, unique_in_group))

        if varying_values:
            warnings[col] = list(varying_values)

    return warnings


def _print_aggregation_warnings(warnings: dict[str, list], figure_index: int) -> None:
    """Print aggregation warnings if any exist."""
    if not warnings:
        return

    print(
        f"\n⚠️  WARNING: Figure #{figure_index} is aggregating over multiple values of:"
    )
    for col, values in sorted(warnings.items()):
        values_str = ", ".join(values[:5])  # Limit to first 5 values
        if len(values) > 5:
            values_str += f", ... ({len(values)} total)"
        print(f"    - {col}: [{values_str}]")
    print("   Consider filtering or grouping by these columns.\n")


# =============================================================================
# Layout Constants
# =============================================================================

# Base font and marker sizes (at scale=1.0)
BASE_LABEL_FONTSIZE = 18
BASE_TICK_FONTSIZE = 15
BASE_LEGEND_FONTSIZE = 15
BASE_BAR_LABEL_FONTSIZE = 15
BASE_MARKER_SIZE = 60
BASE_LINEWIDTH = 2
BASE_FAINT_LINEWIDTH = 1


# =============================================================================
# Display Name Mapping
# =============================================================================

# Columns where subtitles should show just the value, not "column = value"
# Used in Grid.render() when building cell subtitles
VALUE_ONLY_SUBTITLE_COLUMNS: set[str] = {"env_name", "prompter_name", "prompt_label", "env_and_prompter", "hint_and_prompter"}

# Columns that should be hidden from legends when used for grouping (color, marker, etc.)
# These columns can still be used for grouping, they just won't appear in the legend.
LEGEND_HIDDEN_COLUMNS: set[str] = {"baseline_instructions_path"}

# Maps column names to friendly display names for axis labels, titles, etc.
# Names with suffixes (e.g., "hacking_rate_hinted") will match their base name.
COLUMN_DISPLAY_NAMES: dict[str, str] = {
    "discovery_eval_counts": "Task LM Calls",
    "reflection_call_count": "Reflection LM Calls",
    "column_name": "Column",
    "column_value": "Value",
    "suggest_hack": "Told to Hack?",
    "is_sanitized": "Sanitized?",
    "hint_type": "Hint Type",
    "run_index": "Run",
    "candidate_index": "Candidate",
    "prompt_type": "Prompt Type",  # Doesn't exist by default
    "relevant_verbalization": "Hacking Monitor Recall",
    "proxy_reward": "Proxy Reward",
    "true_reward": "True Reward",
    "hacking_rate": "Hacking Rate",
    "prompt_verbalizes": "Verbalizes Hacking?",
    "prompt_verbalizes_float": "Hack Verbalization Level",
    "cot_verbalizes_yes": "CoT Verbalizes (Yes)",
    "cot_verbalizes_mistargeted": "CoT Verbalizes (Mistargeted)",
    "cot_verbalizes_unclear": "CoT Verbalizes (Unclear)",
    "cot_verbalizes_no": "CoT Verbalizes (No)",
    "prompt_suspiciousness": "Prompt Suspicious?",
    "cot_suspiciousness_yes": "CoT Suspicious (Yes)",
    "cot_suspiciousness_unclear": "CoT Suspicious (Unclear)",
    "cot_suspiciousness_no": "CoT Suspicious (No)",
    "stage": "",
    # Shortening columns
    "prompt_length": "Prompt Length (chars)",
    "test_score": "Proxy Reward on Test Set",
    "val_score": "Proxy Reward on Validation Set",
}

# Maps (column, value) pairs to friendly display names for legends, tick labels, etc.
# For "column_name", values with suffixes will match their base name.
VALUE_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "prompter_name": {
        "anthropic/claude-opus-4-5": "Claude Opus 4.5",
        "openai/gpt-5.2": "GPT-5.2",
        "openrouter/google/gemini-3-pro-preview": "Gemini 3.0 Pro",
    },
    "env_name": {
        "psychosis": "Targeted Sycophancy",
        "wordchain": "Word Chain",
        "mcq": "Hinted MMLU",
    },
    "column_name": {
        "proxy_reward": "Proxy Reward",
        "true_reward": "True Reward",
        "hacking_rate": "Hacking Rate",
        "prompt_verbalizes": "Verbalizes Hacking?",
        "prompt_verbalizes_binary": "Verbalizes Hacking?",
        "cot_verbalizes_yes": "CoT Verbalizes (Yes)",
        "cot_verbalizes_mistargeted": "CoT Verbalizes (Mistargeted)",
        "cot_verbalizes_unclear": "CoT Verbalizes (Unclear)",
        "cot_verbalizes_no": "CoT Verbalizes (No)",
        "prompt_suspiciousness": "Prompt Suspicious?",
        "cot_suspiciousness_yes": "CoT Suspicious (Yes)",
        "cot_suspiciousness_unclear": "CoT Suspicious (Unclear)",
        "cot_suspiciousness_no": "CoT Suspicious (No)",
        "val_score": "Validation Proxy Reward",
        "test_score": "Test Proxy Reward",
        "verbalizes_yes_hacking": "Prompt + CoT",
        "verbalizes_prompt_yes_hacking": "Prompt Only",
    },
    "optimizer_type": {
        "GEPA": "GEPA",
        "RL": "RL",
    },
    "is_sanitized": {
        False: "Original Prompt",
        True: "Sanitized Prompt",
    },
    "source": {
        "initial": "Original Prompt",
        "proposed": "Shortened Prompt",
    },
    "prompt_type": {
        "baseline": "Baseline",
        "no_hack": "No Hack",
        # Using newlines in display name for now, revisit if needed
        "no_hack_sanitized": "No Hack\n(sanitized)",
        "explicit_hack": "Hack",
        "explicit_hack_sanitized": "Hack\n(sanitized)",
    },
    "suggest_hack": {
        "explicit": "Hack",
        "no": "No Hack",
    },
    "prompt_verbalizes": {
        "no": "No Hacking",
        "unclear": "Unclear",
        "mistargeted": "Hacking (mistargeted)",
        "yes": "Hacking",
    },
}


def get_display_name(column: str) -> str:
    """Get user-friendly display name for a column.

    Args:
        column: Internal column name

    Returns:
        User-friendly display name, or the original column name if no mapping exists.
    """
    # First try exact match
    if column in COLUMN_DISPLAY_NAMES:
        return COLUMN_DISPLAY_NAMES[column]

    # Try prefix match
    for base_name, display_name in COLUMN_DISPLAY_NAMES.items():
        if _matches_base_name(column, base_name):
            return display_name

    return column


def get_display_value(column: str, value: Any) -> str:
    """Get user-friendly display name for a column value.

    Args:
        column: Column the value belongs to
        value: Internal value

    Returns:
        User-friendly display name, or str(value) if no mapping exists.
    """
    column_mappings = VALUE_DISPLAY_NAMES.get(column, {})

    # First try exact match
    if value in column_mappings:
        return column_mappings[value]

    # For column_name, try prefix match on the value
    if column == "column_name" and isinstance(value, str):
        for base_name, display_name in column_mappings.items():
            if _matches_base_name(value, base_name):
                return display_name

    return str(value)


# =============================================================================
# Value Ordering
# =============================================================================

# Defines the canonical order for categorical values.
# Values not in this list will be sorted alphabetically after the listed ones.
VALUE_ORDER: dict[str, list] = {
    "env_name": ["mcq", "psychosis", "wordchain"],
    "prompt_type": [
        "baseline",
        "no_hack",
        "no_hack_sanitized",
        "explicit_hack",
        "explicit_hack_sanitized",
    ],
    "column_name": ["proxy_reward", "true_reward", "prompt_verbalizes", "hacking_rate"],
    "prompt_verbalizes": ["yes", "mistargeted", "unclear", "no"],
    "optimizer": [
        "GEPA",
        "GEPA + Suggest Hacking",
        "GEPA + Suggest Hacking + RL Teacher",
        "GEPA + Suggest Hacking + RL Teacher with CoT",
        "RL",
    ],
    "stage": ["Pre-GEPA", "Post-GEPA", "Post-GEPA, Sanitized"],
    "is_sanitized": [False, True],
    "suggest_hack": ["no", "explicit"],
}


def sort_values(column: str, values: list) -> list:
    """Sort values according to VALUE_ORDER, with unlisted values at the end.

    Args:
        column: Column name to look up ordering for
        values: List of values to sort

    Returns:
        Sorted list of values
    """
    if column not in VALUE_ORDER:
        # Special case for prompt_label: sort by env name prefix
        if column == "prompt_label":
            return _sort_prompt_labels(values)
        return sorted(values)

    order = VALUE_ORDER[column]
    order_map = {v: i for i, v in enumerate(order)}

    def sort_key(v):
        if v in order_map:
            return (0, order_map[v])
        return (1, str(v))  # Unlisted values go at end, sorted alphabetically

    return sorted(values, key=sort_key)


def _sort_prompt_labels(values: list) -> list:
    """Sort prompt_label values by env name prefix using env_name ordering."""
    # Map env display names to their sort order (from VALUE_ORDER["env_name"])
    env_order = VALUE_ORDER.get("env_name", [])
    env_display_order = [
        VALUE_DISPLAY_NAMES.get("env_name", {}).get(env, env) for env in env_order
    ]
    # Build prefix -> order map
    prefix_order = {prefix: i for i, prefix in enumerate(env_display_order)}

    def sort_key(v):
        # Find which env prefix this label starts with
        for prefix, order in prefix_order.items():
            if str(v).startswith(prefix):
                # Return (env_order, rest of string for secondary sort)
                return (order, str(v))
        return (len(prefix_order), str(v))  # Unknown prefixes go last

    return sorted(values, key=sort_key)


# =============================================================================
# Style Registry
# =============================================================================

# Maps (column, value) pairs to style attributes that should always be used.
# When plotting, if a value has a pinned style, it overrides automatic assignment.
# Supported style keys: "color", "linestyle", "marker"
STYLE_REGISTRY: dict[tuple[str, Any], dict[str, Any]] = {
    # Colors for column names (used when melting columns)
    ("column_name", "proxy_reward"): {"color": "#e69f00"},  # orange
    ("column_name", "true_reward"): {"color": "#0072b2"},  # blue
    ("column_name", "prompt_verbalizes"): {"color": "#cc79a7"},  # muted pink
    ("column_name", "hacking_rate"): {"color": "#d55e00"},  # vermillion
    # Colors for prompt_verbalizes values (semantic: no=safe, yes=hacking)
    ("prompt_verbalizes", "no"): {"color": "#0072b2"},  # blue
    ("prompt_verbalizes", "unclear"): {"color": "#f0e442"},  # yellow
    ("prompt_verbalizes", "mistargeted"): {"color": "#cc79a7"},  # muted pink
    ("prompt_verbalizes", "yes"): {"color": "#d55e00"},  # vermillion
    # Line styles for is_sanitized
    ("is_sanitized", False): {"linestyle": "-"},
    ("is_sanitized", True): {"linestyle": ":"},
    # Markers for shortening source types
    ("source", "initial"): {"marker": "*"},
    ("source", "proposed"): {"marker": "o"},
    ("source", "empty"): {"marker": "o"},
    # Colors for optimizer values (blue progression for GEPA variants, gray for RL baseline)
    ("optimizer", "GEPA"): {"color": "#c6dbef"},  # lightest blue
    ("optimizer", "GEPA + Suggest Hacking"): {"color": "#6baed6"},  # light blue
    ("optimizer", "GEPA + Suggest Hacking + RL Teacher"): {"color": "#2171b5"},  # medium blue
    ("optimizer", "GEPA + Suggest Hacking + RL Teacher with CoT"): {"color": "#084594"},  # dark blue
    ("optimizer", "RL"): {"color": "#555555"},  # dark gray (baseline comparison)
    # Colors for verbalization_label (used in recall_by_hacking_bins plot)
    ("verbalization_label", "GEPA (prompt verbalization)"): {"color": "#30b0ff"},  # bright blue
    ("verbalization_label", "RL (CoT verbalization)"): {"color": "#555555"},  # dark gray (matches RL)
}


def get_style(column: str, value: Any, style_key: str) -> Any | None:
    """Get a pinned style attribute for a (column, value) pair.

    Args:
        column: Column name
        value: Column value
        style_key: Style attribute to retrieve ("color", "linestyle", "marker")

    Returns:
        The pinned style value, or None if no pinned style exists.
    """
    key = (column, value)
    if key in STYLE_REGISTRY:
        return STYLE_REGISTRY[key].get(style_key)
    return None


# Default style cycles for each style type
_STYLE_DEFAULTS = {
    "color": list(plt.cm.tab10.colors),
    "linestyle": ["-", ":", "--", "-."],
    "marker": ["o", "s", "^", "D", "v", "p"],
}


def _get_style_with_fallback(
    column: str, value: Any, style_key: str, fallback_index: int
) -> Any:
    """Get style for a (column, value) pair, with fallback to default cycle.

    Args:
        column: Column name
        value: Column value
        style_key: Style attribute ("color", "linestyle", "marker")
        fallback_index: Index into default cycle if no pinned style exists

    Returns:
        Style value from registry or default cycle
    """
    pinned = get_style(column, value, style_key)
    if pinned is not None:
        return pinned
    defaults = _STYLE_DEFAULTS[style_key]
    return defaults[fallback_index % len(defaults)]


def get_color(column: str, value: Any, fallback_index: int) -> str:
    """Get color for a (column, value) pair, with fallback to default palette."""
    return _get_style_with_fallback(column, value, "color", fallback_index)


def get_linestyle(column: str, value: Any, fallback_index: int) -> str:
    """Get line style for a (column, value) pair, with fallback to default cycle."""
    return _get_style_with_fallback(column, value, "linestyle", fallback_index)


def get_marker(column: str, value: Any, fallback_index: int) -> str:
    """Get marker style for a (column, value) pair, with fallback to default cycle."""
    return _get_style_with_fallback(column, value, "marker", fallback_index)


# =============================================================================
# Utility Functions
# =============================================================================


def _filter_main(
    dfs: DFs, filter: str | Callable[[pd.DataFrame], pd.Series] | None
) -> DFs:
    """Filter only the 'main' DataFrame, passing 'gepa_runs' through unchanged.

    Args:
        dfs: Dict with "main" and "gepa_runs" DataFrames
        filter: None (no filter), callable returning boolean Series, or query string

    Returns:
        New DFs dict with filtered "main" and unchanged "gepa_runs"
    """
    result = {}
    for key, df in dfs.items():
        if key == "main":
            if filter is None:
                result[key] = df
            elif callable(filter):
                result[key] = df[filter(df)]
            else:
                result[key] = df.query(filter)
        elif key == "gepa_runs":
            result[key] = df
        else:
            raise ValueError(f"Unknown DataFrame key: {key}")
    return result


def _melt_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Melt specified columns into column_name/column_value format.

    Args:
        df: Wide-format DataFrame
        columns: List of columns to melt

    Returns:
        Long-format DataFrame with 'column_name' and 'column_value' columns.
        Rows where column_value is NaN are dropped.
    """
    # Find which columns actually exist in the DataFrame
    existing_columns = [c for c in columns if c in df.columns]
    if not existing_columns:
        raise ValueError(f"No columns found in DataFrame. Expected: {columns}")

    # All other columns become id_vars
    id_vars = [col for col in df.columns if col not in columns]

    melted = df.melt(
        id_vars=id_vars,
        value_vars=existing_columns,
        var_name="column_name",
        value_name="column_value",
    )

    # Drop rows where column_value is NaN
    melted = melted.dropna(subset=["column_value"])

    return melted


# =============================================================================
# Layout Nodes (composable plotting components)
# =============================================================================


class GridSize(NamedTuple):
    """Size information returned by LayoutNode.get_grid_size().

    Attributes:
        rows: Total rows in grid units
        cols: Total columns in grid units
        cell_width: Width of one grid cell in inches
        cell_height: Height of one grid cell in inches
    """

    rows: int
    cols: int
    cell_width: float
    cell_height: float


# Default plot dimensions in inches
DEFAULT_PLOT_WIDTH = 4.5
DEFAULT_PLOT_HEIGHT = 3


class LegendEntry(NamedTuple):
    """A single legend entry with metadata for sorting.

    Attributes:
        handle: Matplotlib artist for the legend
        label: Display label for the legend
        column: Original column name (for VALUE_ORDER sorting)
        value: Original value (for VALUE_ORDER sorting)
    """

    handle: Any
    label: str
    column: str
    value: Any


class RenderResult(NamedTuple):
    """Result returned by LayoutNode.render().

    Attributes:
        warnings: Dict of aggregation warnings (column -> unique values)
        legend_entries: List of LegendEntry for figure-level legend
    """

    warnings: dict[str, list]
    legend_entries: list[LegendEntry]

    @staticmethod
    def merge(results: list["RenderResult"]) -> "RenderResult":
        """Merge multiple RenderResults, deduplicating legend entries by label.

        Args:
            results: List of RenderResult objects to merge

        Returns:
            Combined RenderResult with merged warnings and deduplicated legend entries
        """
        all_warnings = defaultdict(set)
        all_legend_entries = []
        seen_labels = set()

        for result in results:
            for col, vals in result.warnings.items():
                all_warnings[col].update(vals)
            for entry in result.legend_entries:
                if entry.label not in seen_labels:
                    seen_labels.add(entry.label)
                    all_legend_entries.append(entry)

        # Sort entries by VALUE_ORDER to ensure consistent ordering
        # regardless of which subplot first contributed each entry
        def sort_key(entry: LegendEntry):
            # Special entries (column starting with "_") go at the end
            is_special = entry.column.startswith("_") if entry.column else False
            # Get the sort position for this value within its column
            order = VALUE_ORDER.get(entry.column, [])
            if entry.value in order:
                value_pos = order.index(entry.value)
            else:
                value_pos = len(order)  # Unlisted values go at end
            # Group by: special flag, column, value order within column
            return (is_special, entry.column, value_pos, str(entry.value))

        all_legend_entries.sort(key=sort_key)

        return RenderResult(
            warnings={col: list(vals) for col, vals in all_warnings.items()},
            legend_entries=all_legend_entries,
        )


class LayoutNode(ABC):
    """Base class for composable layout nodes."""

    @abstractmethod
    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        """Return grid size information for this layout.

        Returns:
            GridSize with total grid dimensions (rows, cols) and cell size in inches
            (cell_width, cell_height). The figure size is computed as cols * cell_width
            by rows * cell_height.
        """
        pass

    @abstractmethod
    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        """Render this layout into the given GridSpec region.

        Args:
            fig: Matplotlib figure
            gs: GridSpec to render into
            row_slice: Row range in the grid
            col_slice: Column range in the grid
            dfs: Dict with "main" (candidate rows) and "gepa_runs" (run-level data)
            scale: Scale factor for fonts/markers (1.0 = full size, <1.0 = smaller)
            show_chrome: Whether to show axis labels and legend
            subtitle: Optional subtitle passed from parent Grid (shows groupby value)
            is_leftmost: Whether this cell is in the leftmost column of the grid
            is_bottommost: Whether this cell is in the bottommost row of the grid

        Returns:
            RenderResult with aggregation warnings and legend entries.
        """
        pass


@dataclass
class Grid(LayoutNode):
    """Creates a grid of plots, one per unique value of groupby column.

    Args:
        groupby: Column to group data by (one plot per unique value)
        cols_wrap: Maximum number of columns before wrapping to next row
        inner: Layout node to render for each group
        shared_axes: If True, only leftmost plots show y-axis labels and only
            bottommost plots show x-axis labels. If False, all plots show their
            own axis labels. Set to False when x/y ranges differ across groups.
    """

    groupby: str
    cols_wrap: int = 3
    inner: LayoutNode = None
    shared_axes: bool = True

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        unique_values = df[self.groupby].dropna().unique()
        n_groups = len(unique_values)

        if self.inner is None:
            inner_size = GridSize(1, 1, DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
        else:
            inner_size = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)
        grid_rows = ceil(n_groups / self.cols_wrap)

        return GridSize(
            rows=grid_rows * inner_size.rows,
            cols=grid_cols * inner_size.cols,
            cell_width=inner_size.cell_width,  # Pass through from inner
            cell_height=inner_size.cell_height,
        )

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        df = dfs["main"]
        unique_values = sort_values(self.groupby, list(df[self.groupby].dropna().unique()))
        n_groups = len(unique_values)

        if self.inner is None:
            inner_size = GridSize(1, 1, DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
        else:
            inner_size = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)
        grid_rows = ceil(n_groups / self.cols_wrap)
        child_results = []

        for i, value in enumerate(unique_values):
            grid_row = i // grid_cols
            grid_col = i % grid_cols

            # Calculate subregion for this group
            r_start = row_slice.start + grid_row * inner_size.rows
            r_end = r_start + inner_size.rows
            c_start = col_slice.start + grid_col * inner_size.cols
            c_end = c_start + inner_size.cols

            sub_dfs = _filter_main(dfs, lambda df: df[self.groupby] == value)

            # Build subtitle showing the groupby column and value
            # Format: "column = value" or just "value" for VALUE_ONLY_SUBTITLE_COLUMNS
            display_value = get_display_value(self.groupby, value)
            if self.groupby in VALUE_ONLY_SUBTITLE_COLUMNS:
                this_level = display_value
            else:
                this_level = f"{self.groupby} = {display_value}"
            cell_subtitle = f"{subtitle}, {this_level}" if subtitle else this_level

            # Determine position flags for this cell
            # When shared_axes=False, all plots show their own axis labels
            if self.shared_axes:
                cell_is_leftmost = is_leftmost and (grid_col == 0)
                cell_is_bottommost = is_bottommost and (grid_row == grid_rows - 1)
            else:
                cell_is_leftmost = True
                cell_is_bottommost = True

            if self.inner is None:
                # No inner layout - just create a placeholder
                ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
                ax.set_title(cell_subtitle)
                ax.text(0.5, 0.5, "No inner layout", ha="center", va="center")
            else:
                # Grid passes scale and show_chrome through unchanged
                result = self.inner.render(
                    fig,
                    gs,
                    slice(r_start, r_end),
                    slice(c_start, c_end),
                    sub_dfs,
                    scale,
                    show_chrome,
                    subtitle=cell_subtitle,
                    is_leftmost=cell_is_leftmost,
                    is_bottommost=cell_is_bottommost,
                )
                child_results.append(result)

        return RenderResult.merge(child_results)


@dataclass
class MainSide(LayoutNode):
    """Creates a main plot with smaller side plots for different groupby values.

    The main plot and side plots scale uniformly: with n side plots, each side
    is 1/n the size of the main in both dimensions.
    """

    groupby: str
    main_value: Any = None  # Which value gets the main (big) plot
    inner: LayoutNode = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        unique_values = df[self.groupby].unique()
        # Filter out NaN if main_value is None (since we compare with ==)
        if self.main_value is None:
            side_values = [v for v in unique_values if pd.notna(v)]
        else:
            side_values = [v for v in unique_values if v != self.main_value]

        n_sides = max(len(side_values), 1)

        if self.inner is None:
            inner_size = GridSize(1, 1, DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
        else:
            inner_size = self.inner.get_grid_size(df)

        # Main plot: n_sides cols × n_sides rows
        # Side plots: 1 col × 1 row each, stacked vertically
        # This gives uniform 1/n_sides scaling in both dimensions for sides
        total_rows = n_sides * inner_size.rows
        total_cols = (n_sides + 1) * inner_size.cols  # n_sides for main, 1 for sides

        # Cell size is inner size / n_sides so main plot (n_sides × n_sides) gets full inner size
        cell_width = inner_size.cell_width / n_sides
        cell_height = inner_size.cell_height / n_sides

        return GridSize(total_rows, total_cols, cell_width, cell_height)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        df = dfs["main"]
        unique_values = list(df[self.groupby].unique())

        # Separate main and side values
        if self.main_value is None:
            # main_value=None means use rows where groupby column is NaN
            main_dfs = _filter_main(dfs, lambda df: df[self.groupby].isna())
            side_values = sort_values(self.groupby, [v for v in unique_values if pd.notna(v)])
        else:
            main_dfs = _filter_main(dfs, lambda df: df[self.groupby] == self.main_value)
            side_values = sort_values(
                self.groupby, [v for v in unique_values if v != self.main_value and pd.notna(v)]
            )

        n_sides = max(len(side_values), 1)

        if self.inner is None:
            inner_size = GridSize(1, 1, DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
        else:
            inner_size = self.inner.get_grid_size(df)

        child_results = []

        # Main plot region: left n_sides/(n_sides+1), full height
        main_col_end = col_slice.start + n_sides * inner_size.cols

        if self.inner is None:
            ax_main = fig.add_subplot(gs[row_slice, col_slice.start : main_col_end])
            ax_main.set_title("Main (no inner layout)")
        else:
            # Main plot gets the current scale and show_chrome unchanged
            # Pass through subtitle from parent
            result = self.inner.render(
                fig,
                gs,
                row_slice,
                slice(col_slice.start, main_col_end),
                main_dfs,
                scale,
                show_chrome,
                subtitle=subtitle,
                is_leftmost=is_leftmost,
                is_bottommost=is_bottommost,
            )
            child_results.append(result)

        # Side plots: right 1/(n_sides+1), stacked vertically
        # Each side is 1/n_sides the linear size of main, so scale by 1/n_sides
        side_scale = scale / n_sides

        for i, side_value in enumerate(side_values):
            side_dfs = _filter_main(dfs, lambda df: df[self.groupby] == side_value)

            r_start = row_slice.start + i * inner_size.rows
            r_end = r_start + inner_size.rows

            if self.inner is None:
                ax_side = fig.add_subplot(
                    gs[r_start:r_end, main_col_end : col_slice.stop]
                )
                ax_side.set_title(f"{self.groupby}={side_value}")
            else:
                # Side plots don't show chrome
                result = self.inner.render(
                    fig,
                    gs,
                    slice(r_start, r_end),
                    slice(main_col_end, col_slice.stop),
                    side_dfs,
                    side_scale,
                    show_chrome=False,
                    is_leftmost=False,
                    is_bottommost=False,
                )
                child_results.append(result)

        return RenderResult.merge(child_results)




@dataclass
class BarPlot(LayoutNode):
    """Bar chart comparing values across x categories.

    Args:
        x: Column for x-axis categories
        y: Column(s) for y-axis values. If a list, melts those columns into
           column_name/column_value and uses column_value for y.
        hue: Column for color grouping. When y is a list, this should typically
             be "column_name" to color by the melted column names.
        title: Optional title for the plot
        width: Plot width in inches (default: 9)
        height: Plot height in inches (default: 6)

    Examples:
        # Compare metrics across hack settings (metrics as different colors)
        BarPlot(x="suggest_hack", y=["proxy_reward", "true_reward"], hue="column_name")

        # Compare metrics on x-axis, colored by hack setting
        BarPlot(x="column_name", y=["proxy_reward", "true_reward"], hue="suggest_hack")

        # Single column, no melt
        BarPlot(x="suggest_hack", y="proxy_reward", hue="executor_name")
    """

    x: str
    y: str | list[str]
    width: float
    height: float
    hue: str = None
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        return GridSize(1, 1, self.width, self.height)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        df = dfs["main"]

        # Handle y as list: melt and use column_value for y
        if isinstance(self.y, list):
            if len(self.y) < 2:
                raise ValueError(
                    "y must have at least 2 columns when specified as a list"
                )
            df = _melt_columns(df, self.y)
            y_col = "column_value"
            # Validate that column_name is used somewhere
            if "column_name" not in {self.x, self.hue}:
                raise ValueError(
                    "When y is a list, 'column_name' must be used for x or hue"
                )
        else:
            y_col = self.y

        # Use subtitle from parent Grid if no explicit title set
        title = self.title if self.title is not None else subtitle

        ax = fig.add_subplot(gs[row_slice, col_slice])
        legend_entries = _draw_bar_plot(
            ax, df, self.x, y_col, self.hue, title, scale, show_chrome,
            is_leftmost=is_leftmost, is_bottommost=is_bottommost
        )
        warnings = _check_aggregation_warnings(df, self.x, y_col, self.hue, type(self))
        return RenderResult(warnings=warnings, legend_entries=legend_entries)


@dataclass
class ProgressionPlot(LayoutNode):
    """Line chart showing value progression over time.

    Args:
        x: Column for x-axis values (typically discovery_eval_counts)
        y: Column(s) for y-axis values. If a list, melts those columns into
           column_name/column_value and uses column_value for y.
        hue: Column for color grouping. When y is a list, this should typically
             be "column_name" to color by the melted column names.
        linestyle: Column for line style grouping
        hue_as_hline: List of hue values to draw as horizontal lines instead of
            progressions. For these hue values, the mean y-value is computed and
            drawn as a dashed horizontal line. Useful for showing RL baselines
            that don't have progression data.
        title: Optional title for the plot
        show_individual_lines: Whether to show faint lines for individual runs
        width: Plot width in inches (default: 9)
        height: Plot height in inches (default: 6)

    Examples:
        # Compare metrics over time (metrics as different colors)
        ProgressionPlot(x="discovery_eval_counts",
                        y=["proxy_reward", "true_reward"], hue="column_name")

        # Metrics as colors, linestyle by sanitized
        ProgressionPlot(x="discovery_eval_counts",
                        y=["proxy_reward", "true_reward"], hue="column_name",
                        linestyle="is_sanitized")

        # Single column, compare by suggest_hack
        ProgressionPlot(x="discovery_eval_counts", y="proxy_reward", hue="suggest_hack")

        # With RL baseline as horizontal line (data comes from hue="optimizer" where optimizer="RL")
        ProgressionPlot(x="discovery_eval_counts", y="proxy_reward", hue="optimizer",
                        hue_as_hline=["RL"])
    """

    width: float
    height: float
    x: str = "discovery_eval_counts"
    y: str | list[str] = None  # Required, but default None for backwards compat error
    hue: str = None
    linestyle: str = None
    hue_as_hline: list[str] = None
    title: str = None
    show_individual_lines: bool = True
    show_max_star: bool = False

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        return GridSize(1, 1, self.width, self.height)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        if self.y is None:
            raise ValueError("ProgressionPlot requires 'y' parameter")

        df = dfs["main"]

        # Handle y as list: melt and use column_value for y
        if isinstance(self.y, list):
            if len(self.y) < 2:
                raise ValueError(
                    "y must have at least 2 columns when specified as a list"
                )
            df = _melt_columns(df, self.y)
            y_col = "column_value"
            # Validate that column_name is used somewhere
            if "column_name" not in {self.x, self.hue, self.linestyle}:
                raise ValueError(
                    "When y is a list, 'column_name' must be used for x, hue, or linestyle"
                )
        else:
            y_col = self.y

        plot_dfs = {"main": df, "gepa_runs": dfs["gepa_runs"]}

        # Use subtitle from parent Grid if no explicit title set
        title = self.title if self.title is not None else subtitle

        ax = fig.add_subplot(gs[row_slice, col_slice])
        legend_entries = _draw_progression_plot(
            ax,
            plot_dfs,
            self.x,
            y_col,
            self.hue,
            self.linestyle,
            self.hue_as_hline,
            title,
            scale,
            show_chrome,
            self.show_individual_lines,
            self.show_max_star,
            is_leftmost=is_leftmost,
            is_bottommost=is_bottommost,
        )
        warnings = _check_aggregation_warnings(
            df, self.x, y_col, self.hue, type(self), self.linestyle
        )
        return RenderResult(warnings=warnings, legend_entries=legend_entries)


@dataclass
class ScatterPlot(LayoutNode):
    """Scatter plot comparing two metrics with optional color and marker grouping.

    Args:
        x: Column for x-axis values
        y: Column(s) for y-axis values. If a list, melts those columns into
           column_name/column_value and uses column_value for y, with color
           automatically set to "column_name".
        color: Column for color grouping (optional). When y is a list, this
            defaults to "column_name".
        marker: Column for marker shape grouping (optional)
        group_by: Columns to group by (optional). If None, shows individual points
            (one per row). If specified, shows one point per unique combination
            of the group_by columns, with x/y values being the mean of each group.
        arrow_by: Column for drawing arrows between paired points (optional).
            Must be a boolean column. Draws arrows from False to True points.
            If group_by is specified, arrow_by must be included in group_by.
        pareto_line: Whether to draw Pareto frontier step lines, grouped by color.
        pareto_direction: Dict specifying which direction is "better" for each axis.
            Required when pareto_line is True. Keys are "x" and "y", values are
            "lower" or "higher". E.g., {"x": "lower", "y": "higher"} means lower
            x and higher y are better.
        hide_pareto_dominated: Whether to hide points dominated on the Pareto frontier.
            If True, only Pareto-optimal points are shown, plus any points with marker
            values in show_dominated_markers.
        show_dominated_markers: List of marker values that should always be shown
            even if Pareto-dominated. E.g., ["initial"] to always show initial points.
        title: Optional title for the plot
        width: Plot width in inches (default: 6, square)
        height: Plot height in inches (default: 6, square)

    Examples:
        # Individual points, colored by prompt_verbalizes
        ScatterPlot(x="proxy_reward", y="true_reward", color="prompt_verbalizes")

        # Multiple y columns - each becomes a different colored series
        ScatterPlot(x="prompt_length", y=["val_score", "test_score"])

        # One point per prompt_verbalizes value (average x/y within each group)
        ScatterPlot(x="proxy_reward", y="true_reward", color="prompt_verbalizes",
                    group_by=["prompt_verbalizes"])

        # Arrows from original to sanitized points (no grouping)
        ScatterPlot(x="proxy_reward", y="true_reward", color="is_sanitized",
                    arrow_by="is_sanitized")

        # Pareto frontier with step lines (lower x, higher y is better)
        ScatterPlot(x="prompt_length", y="test_score", color="env_name",
                    pareto_line=True, pareto_direction={"x": "lower", "y": "higher"})
    """

    x: str
    y: str | list[str]
    width: float
    height: float
    color: str = None
    marker: str = None
    group_by: list[str] = None
    arrow_by: str = None
    pareto_line: bool = False
    pareto_direction: dict[str, str] = None
    hide_pareto_dominated: bool = False
    show_dominated_markers: list[str] = None
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        return GridSize(1, 1, self.width, self.height)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        df = dfs["main"]

        # Handle y as list: melt and use column_value for y
        if isinstance(self.y, list):
            if len(self.y) < 2:
                raise ValueError(
                    "y must have at least 2 columns when specified as a list"
                )
            if self.color is not None:
                raise ValueError(
                    "Cannot set color when y is a list (color is used to distinguish metrics)"
                )
            df = _melt_columns(df, self.y)
            y_col = "column_value"
            color_col = "column_name"
        else:
            y_col = self.y
            color_col = self.color

        # Validate group_by includes color/marker if specified
        if self.group_by is not None:
            if color_col is not None and color_col not in self.group_by:
                raise ValueError(
                    f"group_by must include color column '{color_col}'"
                )
            if self.marker is not None and self.marker not in self.group_by:
                raise ValueError(
                    f"group_by must include marker column '{self.marker}'"
                )

        # Validate arrow_by is in group_by when both are specified
        if self.arrow_by is not None and self.group_by is not None:
            if self.arrow_by not in self.group_by:
                raise ValueError(
                    f"arrow_by='{self.arrow_by}' must be included in group_by "
                    f"to preserve pairs for arrow drawing"
                )

        # Validate pareto_direction is provided when pareto_line is True
        if self.pareto_line and self.pareto_direction is None:
            raise ValueError(
                "pareto_direction must be specified when pareto_line=True"
            )

        # Use subtitle from parent Grid if no explicit title set
        title = self.title if self.title is not None else subtitle

        ax = fig.add_subplot(gs[row_slice, col_slice])
        legend_entries = _draw_scatter_plot(
            ax,
            df,
            self.x,
            y_col,
            color_col,
            self.marker,
            self.group_by,
            self.arrow_by,
            self.pareto_line,
            self.pareto_direction,
            self.hide_pareto_dominated,
            self.show_dominated_markers,
            title,
            scale,
            show_chrome,
            is_leftmost=is_leftmost,
            is_bottommost=is_bottommost,
        )
        warnings = _check_aggregation_warnings(
            df, self.x, y_col, color_col, type(self), self.marker
        )
        return RenderResult(warnings=warnings, legend_entries=legend_entries)


@dataclass
class BinnedLinePlot(LayoutNode):
    """Line plot showing binned averages of y values across x bins.

    Useful for showing how a metric (y) varies across ranges of another metric (x),
    with separate lines for different groups (hue).

    Args:
        x: Column for x-axis values (will be binned)
        y: Column(s) for y-axis values (will be averaged within each bin).
           If a list, melts those columns and creates separate lines for each
           (hue_value, column_name) combination.
        hue: Column for color grouping (each unique value becomes a separate line).
             When y is a list, lines are distinguished by (hue_value, column_name).
        linestyle: Column for line style grouping (optional). When y is a list,
             this should typically be "column_name" to distinguish metrics.
        n_bins: Number of bins for x-axis (default 10)
        title: Optional title for the plot
        equal_weight_by: Column to weight equally when averaging (optional).
            When set, computes per-group averages first, then averages those.
            Useful when groups have different amounts of data but should contribute
            equally to the final average (e.g., equal_weight_by="env_name").
        show_error_bars: Whether to show ± SEM as error bars at each point.
            Only supported when equal_weight_by is None. Points with count=1
            won't have error bars (can't compute SEM from a single sample).
        show_counts: Whether to show sample counts as text labels at each point.
        width: Plot width in inches (default: 9)
        height: Plot height in inches (default: 6)

    Examples:
        # Show recall vs hacking rate bins, comparing GEPA vs RL
        BinnedLinePlot(
            x="hacking_rate",
            y="verbalizes_yes_hacking",
            hue="optimizer_type",
            n_bins=10,
        )

        # Compare multiple metrics across hue values
        BinnedLinePlot(
            x="hacking_rate",
            y=["verbalizes_yes_hacking", "verbalizes_prompt_yes_hacking"],
            hue="optimizer_type",
            linestyle="column_name",
            n_bins=10,
        )

        # Weight environments equally (macro average)
        BinnedLinePlot(
            x="hacking_rate",
            y="verbalizes_yes_hacking",
            hue="optimizer_type",
            equal_weight_by="env_name",
            n_bins=10,
        )
    """

    x: str
    y: str | list[str]
    width: float
    height: float
    hue: str = None
    linestyle: str = None
    n_bins: int = 10
    title: str = None
    equal_weight_by: str = None
    show_error_bars: bool = False
    show_counts: bool = False

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        return GridSize(1, 1, self.width, self.height)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        dfs: DFs,
        scale: float = 1.0,
        show_chrome: bool = True,
        subtitle: str = None,
        is_leftmost: bool = True,
        is_bottommost: bool = True,
    ) -> RenderResult:
        df = dfs["main"]

        # Handle y as list: melt and use column_value for y
        if isinstance(self.y, list):
            if len(self.y) < 2:
                raise ValueError(
                    "y must have at least 2 columns when specified as a list"
                )
            df = _melt_columns(df, self.y)
            y_col = "column_value"
            # Validate that column_name is used somewhere
            if "column_name" not in {self.hue, self.linestyle}:
                raise ValueError(
                    "When y is a list, 'column_name' must be used for hue or linestyle"
                )
        else:
            y_col = self.y

        # Use subtitle from parent Grid if no explicit title set
        title = self.title if self.title is not None else subtitle

        ax = fig.add_subplot(gs[row_slice, col_slice])
        legend_entries = _draw_binned_line_plot(
            ax, df, self.x, y_col, self.hue, self.linestyle, self.n_bins, title, scale, show_chrome,
            self.equal_weight_by, self.show_error_bars, self.show_counts,
            is_leftmost=is_leftmost, is_bottommost=is_bottommost
        )
        # No aggregation warnings for binned plots (binning is the point)
        return RenderResult(warnings={}, legend_entries=legend_entries)


# =============================================================================
# Computed Column Helpers
# =============================================================================


def make_prompt_type_column(df: pd.DataFrame) -> pd.Series:
    """Create a 'prompt_type' column with 5 categories for bar plot comparison.

    Categories (in order):
        1. "baseline" - First candidate (is_baseline=True), not sanitized
        2. "no_hack" - Final candidate with suggest_hack="no", not sanitized
        3. "no_hack_sanitized" - Final candidate with suggest_hack="no", sanitized
        4. "explicit_hack" - Final candidate with suggest_hack="explicit", not sanitized
        5. "explicit_hack_sanitized" - Final candidate with suggest_hack="explicit", sanitized

    Rows that don't match any category get None (and should be filtered out).

    Usage:
        PlotConfig(
            computed_columns={"prompt_type": make_prompt_type_column},
            figures=[
                Figure(
                    name="Scores by prompt type",
                    filter=lambda df: df["prompt_type"].notna(),
                    layout=BarPlot(x="prompt_type", y=["proxy_reward", "true_reward"], hue="column_name"),
                ),
            ],
        )
    """

    def classify_row(row):
        # Baseline: first candidate, not sanitized
        # Ignore the unsanitized baseline, since it should be the same
        if row["is_baseline"] and not row["is_sanitized"]:
            return "baseline"

        # Final candidates only for the rest
        if not row["is_final"]:
            return None

        hack = row["suggest_hack"]
        is_sanitized = row["is_sanitized"]

        if hack == "no" and not is_sanitized:
            return "no_hack"
        elif hack == "no" and is_sanitized:
            return "no_hack_sanitized"
        elif hack == "explicit" and not is_sanitized:
            return "explicit_hack"
        elif hack == "explicit" and is_sanitized:
            return "explicit_hack_sanitized"

        return None

    return df.apply(classify_row, axis=1)


def make_prompt_verbalizes_column(df: pd.DataFrame) -> pd.Series:
    """Convert verbalizes_prompt_* float columns to a single string column.

    Takes the 4 verbalizes_prompt_* columns (yes, mistargeted, unclear, no) which
    contain fractions, and returns the category with the highest fraction.

    Tie-breaking priority: yes > mistargeted > unclear > no

    Args:
        df: DataFrame with verbalizes_prompt_yes, verbalizes_prompt_mistargeted,
            verbalizes_prompt_unclear, verbalizes_prompt_no columns

    Returns:
        Series with string values: "yes", "mistargeted", "unclear", or "no"
    """
    # Priority order for tie-breaking (higher priority first)
    categories = ["yes", "mistargeted", "unclear", "no"]
    columns = [f"verbalizes_prompt_{cat}" for cat in categories]

    # Check that at least one column exists
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        raise ValueError(
            f"DataFrame must have at least one of {columns}. "
            f"Available columns: {list(df.columns)}"
        )

    def get_max_category(row):
        max_val = -1
        max_cat = None
        for cat, col in zip(categories, columns):
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and val > max_val:
                    max_val = val
                    max_cat = cat
        return max_cat

    return df.apply(get_max_category, axis=1)


# =============================================================================
# Plotting Implementation
# =============================================================================


def _draw_bar_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
    is_leftmost: bool = True,
    is_bottommost: bool = True,
) -> list[LegendEntry]:
    """Draw a bar plot with individual data points overlaid.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame with data
        x: Column for x-axis categories
        y: Column for y-axis values (bar heights)
        hue: Column for color grouping (can be None for single-color bars)
        title: Optional title for the plot
        scale: Scale factor for fonts and markers (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
        is_leftmost: Whether this plot is in the leftmost column of the grid
        is_bottommost: Whether this plot is in the bottommost row of the grid

    Returns:
        List of (handle, label) tuples for legend entries
    """
    legend_entries = []

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return legend_entries

    # Get unique x values, sorted by canonical order
    x_values = sort_values(x, list(df[x].unique()))

    # Handle optional hue
    hue_values = sort_values(hue, list(df[hue].unique())) if hue is not None else [None]

    n_x = len(x_values)
    n_hue = len(hue_values)

    if n_x == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return legend_entries

    # Scale font and marker sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    bar_label_fontsize = BASE_BAR_LABEL_FONTSIZE * scale
    marker_size = BASE_MARKER_SIZE * scale

    # Calculate bar positions
    bar_width = 0.8 / n_hue
    x_positions = np.arange(n_x)

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val] if hue_val is not None else df

        # Calculate mean and individual values for each x category
        means = []
        all_individuals = []

        for x_val in x_values:
            subset = hue_df[hue_df[x] == x_val][y]
            means.append(subset.mean() if len(subset) > 0 else 0)
            all_individuals.append(subset.values)

        # Get color and label
        color = get_color(hue, hue_val, i)
        display_label = get_display_value(hue, hue_val) if hue_val is not None else None

        # Draw bars
        bar_positions = x_positions + (i - n_hue / 2 + 0.5) * bar_width
        bars = ax.bar(bar_positions, means, bar_width, label=display_label, color=color)

        # Collect legend entry
        if display_label is not None:
            legend_entries.append(LegendEntry(bars[0], display_label, hue, hue_val))

        # Add bar labels
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=bar_label_fontsize)

        # Overlay individual data points with jittered positions
        # Create darker version of bar color for dots
        rgb = mcolors.to_rgb(color)
        dot_color = tuple(c * 0.6 for c in rgb)  # darken by 40%
        for j, (pos, individuals) in enumerate(zip(bar_positions, all_individuals)):
            if len(individuals) > 0:
                # Use deterministic seed based on position for reproducibility
                rng = np.random.default_rng(seed=i * 1000 + j)
                jitter_width = bar_width * 0.6
                jitter = rng.uniform(-jitter_width / 2, jitter_width / 2, len(individuals))
                ax.scatter(
                    pos + jitter,
                    individuals,
                    marker="o",
                    color=dot_color,
                    s=2,
                    alpha=0.7,
                    zorder=3,
                )

    ax.set_xticks(x_positions)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Show x-axis labels and ticks only for bottommost plots
    if show_chrome and is_bottommost:
        ax.set_xticklabels(
            [get_display_value(x, v) for v in x_values], fontsize=tick_fontsize
        )
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
    else:
        ax.set_xticklabels([])

    # Show y-axis labels and ticks only for leftmost plots
    if show_chrome and is_leftmost:
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if title:
        ax.set_title(title, fontsize=label_fontsize)

    return legend_entries


def _draw_progression_series(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    run_end_xs: dict,
    color: str,
    ls: str,
    label: str,
    linewidth: float,
    faint_linewidth: float,
    show_individual_lines: bool = True,
) -> Any:
    """Draw a single progression series (faint individual lines + bold average).

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame filtered to a single hue/linestyle combination
        x: Column for x-axis
        y: Column for y-axis values
        run_end_xs: Dict mapping (experiment_path, run_index) to end x-value (for line extension)
        color: Line color
        ls: Line style
        label: Legend label
        linewidth: Width for bold average line
        faint_linewidth: Width for faint individual lines
        show_individual_lines: Whether to show faint lines for individual runs

    Returns:
        The Line2D object for the average line, or None if no data
    """
    all_run_lines = []

    # Use (experiment_path, run_index) to uniquely identify runs
    for (exp_path, run_idx), run_df in df.groupby(["experiment_path", "run_index"]):
        if run_df.empty:
            continue
        run_df = run_df.sort_values(x)

        end_x = run_end_xs[(exp_path, run_idx)]

        x_vals, y_vals = _build_step_function(
            run_df[x].values, run_df[y].values, end_x=end_x
        )
        if len(x_vals) > 0:
            all_run_lines.append((x_vals, y_vals))
            if show_individual_lines:
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    alpha=0.3,
                    linewidth=faint_linewidth,
                    linestyle=ls,
                )

    if all_run_lines:
        avg_x, avg_y = _average_step_functions(all_run_lines)
        (line,) = ax.plot(
            avg_x,
            avg_y,
            color=color,
            alpha=1.0,
            linewidth=linewidth,
            linestyle=ls,
            label=label,
        )
        return line
    return None


def _draw_hline_series(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    hue_val: str,
    color: str,
    linewidth: float,
) -> Any:
    """Draw a horizontal line for a hue value (e.g., RL baseline).

    Validates that all rows have the same x value, then draws a dashed
    horizontal line at the mean y value.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame filtered to just this hue value
        x: Column for x-axis (used for validation)
        y: Column for y-axis values
        hue: Name of hue column (for display)
        hue_val: The hue value being drawn
        color: Line color
        linewidth: Line width

    Returns:
        The Line2D object for the horizontal line
    """
    # Validate that all x values are the same
    unique_x = df[x].unique()
    if len(unique_x) > 1:
        raise ValueError(
            f"hue_as_hline='{hue_val}' has multiple {x} values: {sorted(unique_x)}. "
            f"Expected all rows to have the same {x} value for horizontal line drawing."
        )

    # Compute mean y-value and draw horizontal line
    mean_y = df[y].mean()
    label = get_display_value(hue, hue_val)
    line = ax.axhline(
        y=mean_y,
        color=color,
        linestyle="--",
        linewidth=linewidth,
        alpha=0.8,
        label=label,
        zorder=2,
    )
    return line


def _draw_progression_plot(
    ax: plt.Axes,
    dfs: DFs,
    x: str,
    y: str,
    hue: str,
    linestyle_col: str = None,
    hue_as_hline: list[str] = None,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
    show_individual_lines: bool = True,
    show_max_star: bool = False,
    is_leftmost: bool = True,
    is_bottommost: bool = True,
) -> list[LegendEntry]:
    """Draw a progression plot with step functions.

    Args:
        ax: Matplotlib axes to draw on
        dfs: Dict with "main" (candidate rows) and "gepa_runs" (run-level data)
        x: Column for x-axis
        y: Column for y-axis values
        hue: Column for color grouping (can be None)
        linestyle_col: Column for line style grouping (optional)
        hue_as_hline: List of hue values to draw as horizontal lines instead of
            progressions. For these, the mean y-value is computed and drawn as
            a dashed line.
        title: Optional title for the plot
        scale: Scale factor for fonts and line widths (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
        show_individual_lines: Whether to show faint lines for individual run_index values
        show_max_star: Whether to show star markers at the maximum final score for each hue
        is_leftmost: Whether this plot is in the leftmost column of the grid
        is_bottommost: Whether this plot is in the bottommost row of the grid

    Returns:
        List of (handle, label) tuples for legend entries
    """
    df = dfs["main"]
    gepa_runs = dfs["gepa_runs"]

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return []

    # Map x column to the corresponding end column in gepa_runs
    # Use composite key (experiment_path, run_index) since run_index alone is not globally unique
    end_x_col = f"end_{x}"
    run_end_xs = {}
    for _, row in gepa_runs.iterrows():
        key = (row["experiment_path"], row["run_index"])
        if key in run_end_xs:
            raise ValueError(f"Duplicate run key: {key}")
        run_end_xs[key] = row[end_x_col]

    # Scale font and line sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    linewidth = BASE_LINEWIDTH * scale
    faint_linewidth = BASE_FAINT_LINEWIDTH * scale

    # Handle optional hue and linestyle
    hue_values = sort_values(hue, list(df[hue].unique())) if hue is not None else [None]
    linestyle_values = (
        sort_values(linestyle_col, list(df[linestyle_col].dropna().unique()))
        if linestyle_col is not None
        else [None]
    )

    # Build index maps for consistent styling
    hue_index = {val: i for i, val in enumerate(hue_values)}
    linestyle_index = {val: j for j, val in enumerate(linestyle_values)}

    # Track which values were actually drawn
    drawn_hue_vals = set()
    drawn_linestyle_vals = set()

    # Separate hue values into progression lines vs horizontal lines
    hline_hue_values = set(hue_as_hline) if hue_as_hline else set()

    # Track final y-values for each hue (for show_max_star)
    # Maps hue_val -> list of final y-values (one per run)
    hue_final_y_values: dict[Any, list[float]] = defaultdict(list)

    # Compute global max x across all runs (for star x-position)
    global_max_x = max(run_end_xs.values()) if run_end_xs else 0

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val] if hue_val is not None else df
        color = get_color(hue, hue_val, i)

        # Draw as horizontal line if specified in hue_as_hline
        if hue_val in hline_hue_values:
            line = _draw_hline_series(ax, hue_df, x, y, hue, hue_val, color, linewidth)
            if line is not None and hue_val is not None:
                drawn_hue_vals.add(hue_val)
            continue

        # Collect final y-values for this hue (across all linestyles)
        if show_max_star:
            for (exp_path, run_idx), run_df in hue_df.groupby(["experiment_path", "run_index"]):
                if run_df.empty:
                    continue
                # Get the final y-value (highest x value for this run)
                run_df_sorted = run_df.sort_values(x)
                final_y = run_df_sorted[y].iloc[-1]
                hue_final_y_values[hue_val].append(final_y)

        for j, linestyle_val in enumerate(linestyle_values):
            # Filter by linestyle column if applicable
            series_df = (
                hue_df[hue_df[linestyle_col] == linestyle_val]
                if linestyle_val is not None
                else hue_df
            )

            # Get linestyle
            ls = (
                get_linestyle(linestyle_col, linestyle_val, j)
                if linestyle_val is not None
                else "-"
            )

            line = _draw_progression_series(
                ax,
                series_df,
                x,
                y,
                run_end_xs,
                color,
                ls,
                None,  # No label - we'll build legend separately
                linewidth,
                faint_linewidth,
                show_individual_lines,
            )
            if line is not None:
                if hue_val is not None:
                    drawn_hue_vals.add(hue_val)
                if linestyle_val is not None:
                    drawn_linestyle_vals.add(linestyle_val)

    # Draw star markers at max final score for each hue (if enabled)
    if show_max_star:
        marker_size = BASE_MARKER_SIZE * scale * 3  # Larger stars
        for i, hue_val in enumerate(hue_values):
            if hue_val in hline_hue_values:
                continue  # Skip hline hues
            final_ys = hue_final_y_values.get(hue_val, [])
            if not final_ys:
                continue
            max_final_y = max(final_ys)
            color = get_color(hue, hue_val, i)
            ax.scatter(
                [global_max_x],
                [max_final_y],
                marker="*",
                s=marker_size,
                c=[color],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Show x-axis labels and ticks only for bottommost plots
    if show_chrome and is_bottommost:
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    # Show y-axis labels and ticks only for leftmost plots
    if show_chrome and is_leftmost:
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if title:
        ax.set_title(title, fontsize=label_fontsize)

    # Build legend entries with separate sections for color and linestyle
    non_hline_hue_vals = drawn_hue_vals - hline_hue_values
    legend_entries = _build_style_legend_entries(
        ax,
        style_specs=[
            (hue, non_hline_hue_vals, hue_index, get_color, {}),
            (linestyle_col, drawn_linestyle_vals, linestyle_index, get_linestyle, {"linewidth": linewidth}),
        ],
    )

    # Add legend entries for hline hues as dashed lines (not patches)
    for hue_val in sort_values(hue, list(drawn_hue_vals & hline_hue_values)):
        color = get_color(hue, hue_val, hue_index.get(hue_val, 0))
        (handle,) = ax.plot([], [], linestyle="--", color=color, linewidth=linewidth)
        legend_entries.append(LegendEntry(handle, get_display_value(hue, hue_val), hue, hue_val))

    # Add legend entry for star marker if show_max_star is enabled
    if show_max_star and hue_final_y_values:
        marker_size = BASE_MARKER_SIZE * scale * 3
        star_handle = ax.scatter(
            [], [], marker="*", s=marker_size, c="gray", edgecolors="black", linewidths=0.5
        )
        legend_entries.append(LegendEntry(star_handle, "Best Run", "_star", "_star"))

    return legend_entries


def _is_display_value_missing(column: str, value: Any) -> bool:
    """Check if a value is missing from VALUE_DISPLAY_NAMES for a column that has mappings.

    Returns True if:
    1. The column is a key in VALUE_DISPLAY_NAMES (meaning we have defined mappings for it)
    2. AND the specific value is not found in those mappings

    Returns False if the column is not in VALUE_DISPLAY_NAMES (no mappings defined).
    """
    if column is None or value is None:
        return False

    if column not in VALUE_DISPLAY_NAMES:
        return False

    column_mappings = VALUE_DISPLAY_NAMES[column]

    # Check exact match
    if value in column_mappings:
        return False

    # For column_name, check prefix match
    if column == "column_name" and isinstance(value, str):
        for base_name in column_mappings.keys():
            if _matches_base_name(value, base_name):
                return False

    return True


def _build_style_legend_entries(
    ax: plt.Axes,
    style_specs: list[tuple],
) -> list[LegendEntry]:
    """Build legend handles and labels with separate sections for each style attribute.

    Creates proxy artists for legend entries:
    - Colors use Patch (colored rectangle)
    - Markers use scatter with the actual marker shape
    - Linestyles use Line2D with the actual linestyle

    Values missing from VALUE_DISPLAY_NAMES are skipped with a warning.

    Args:
        ax: Matplotlib axes (used to create scatter/line proxy artists)
        style_specs: List of tuples, each containing:
            - col: Column name for this style attribute
            - vals: Set of unique values for this attribute
            - index: Dict mapping values to indices for style lookup
            - get_style_fn: Function to get style value (get_color, get_marker, get_linestyle)
            - artist_kwargs: Dict of artist attributes (e.g., {"s": size} for scatter,
              {"linewidth": lw} for lines). Ignored for color entries which use Patch.

    Returns:
        List of LegendEntry for legend
    """
    legend_entries = []

    for col, vals, index, get_style_fn, artist_kwargs in style_specs:
        if col is None or not vals:
            continue
        # Skip columns that should be hidden from legend
        if col in LEGEND_HIDDEN_COLUMNS:
            continue
        # Sort values by canonical order
        sorted_vals = sort_values(col, list(vals))
        for val in sorted_vals:
            if _is_display_value_missing(col, val):
                print(
                    f"Warning: Value '{val}' for column '{col}' not found in "
                    f"VALUE_DISPLAY_NAMES, excluding from legend"
                )
                continue
            style = get_style_fn(col, val, index.get(val, 0))

            if get_style_fn == get_color:
                # Color entries use Patch
                handle = Patch(facecolor=style, edgecolor="none")
            elif get_style_fn == get_marker:
                # Marker entries use scatter
                handle = ax.scatter([], [], marker=style, c="black", **artist_kwargs)
            else:
                # Linestyle entries use Line2D
                (handle,) = ax.plot([], [], linestyle=style, color="black", **artist_kwargs)

            legend_entries.append(LegendEntry(handle, get_display_value(col, val), col, val))

    return legend_entries


def _compute_pareto_frontier(
    points: list[tuple[float, float, int]],
    x_better: str,
    y_better: str,
) -> list[int]:
    """Compute indices of points on the Pareto frontier.

    A point is on the Pareto frontier if no other point dominates it.
    Point A dominates point B if A is better than or equal to B on all
    dimensions, and strictly better on at least one.

    Args:
        points: List of (x, y, original_index) tuples
        x_better: "lower" or "higher" - which direction is better for x
        y_better: "lower" or "higher" - which direction is better for y

    Returns:
        List of original indices of points on the Pareto frontier
    """
    frontier_indices = []
    for i, (xi, yi, idx_i) in enumerate(points):
        dominated = False
        for j, (xj, yj, _) in enumerate(points):
            if i == j:
                continue
            # Check if j dominates i (j is at least as good on both, strictly better on one)
            if x_better == "lower":
                x_at_least = xj <= xi
                x_strict = xj < xi
            elif x_better == "higher":
                x_at_least = xj >= xi
                x_strict = xj > xi
            else:
                raise ValueError(f"Invalid x_better: {x_better}")

            if y_better == "lower":
                y_at_least = yj <= yi
                y_strict = yj < yi
            elif y_better == "higher":
                y_at_least = yj >= yi
                y_strict = yj > yi
            else:
                raise ValueError(f"Invalid y_better: {y_better}")

            if x_at_least and y_at_least and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            frontier_indices.append(idx_i)
    return frontier_indices


def _draw_scatter_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str = None,
    marker_col: str = None,
    group_by: list[str] = None,
    arrow_by: str = None,
    pareto_line: bool = False,
    pareto_direction: dict[str, str] = None,
    hide_pareto_dominated: bool = False,
    show_dominated_markers: list[str] = None,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
    is_leftmost: bool = True,
    is_bottommost: bool = True,
) -> list[LegendEntry]:
    """Draw a scatter plot with optional grouping, arrows, and Pareto frontier.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame with data
        x: Column for x-axis values
        y: Column for y-axis values
        color_col: Column for color grouping (optional)
        marker_col: Column for marker shape grouping (optional)
        group_by: Columns to group by (optional). If None, shows individual points.
            If specified, shows one point per unique combination with mean x/y.
        arrow_by: Column for drawing arrows between paired points (optional).
            Must be a boolean column. Draws arrows from False to True points.
        pareto_line: Whether to draw Pareto frontier step lines, grouped by color.
        pareto_direction: Dict with keys "x" and "y", values "lower" or "higher".
        hide_pareto_dominated: Whether to hide dominated points from scatter.
        show_dominated_markers: Marker values to always show even if dominated.
        title: Optional title for the plot
        scale: Scale factor for fonts and markers (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
        is_leftmost: Whether this plot is in the leftmost column of the grid
        is_bottommost: Whether this plot is in the bottommost row of the grid

    Returns:
        List of (handle, label) tuples for legend entries
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return []

    # Scale font and marker sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    marker_size = BASE_MARKER_SIZE * 1 * scale  # Larger markers for scatter

    # Get unique values for color and marker, sorted by canonical order
    color_values = (
        sort_values(color_col, list(df[color_col].dropna().unique()))
        if color_col is not None
        else [None]
    )
    marker_values = (
        sort_values(marker_col, list(df[marker_col].dropna().unique()))
        if marker_col is not None
        else [None]
    )

    # Build color/marker index maps for consistent styling
    color_index = {val: i for i, val in enumerate(color_values)}
    marker_index = {val: j for j, val in enumerate(marker_values)}

    # Prepare data: group and average if requested, otherwise use as-is
    plot_df = (
        df.groupby(group_by, dropna=False).agg({x: "mean", y: "mean"}).reset_index()
        if group_by is not None
        else df.copy()
    )

    # Compute Pareto frontiers per color group if needed
    pareto_indices_by_color = {}  # color_val -> set of row indices that are Pareto-optimal
    if pareto_line or hide_pareto_dominated:
        for color_val in color_values:
            if color_col is not None:
                group_mask = plot_df[color_col] == color_val
            else:
                group_mask = pd.Series(True, index=plot_df.index)

            group_df = plot_df[group_mask]
            if len(group_df) < 2:
                pareto_indices_by_color[color_val] = set(group_df.index)
                continue

            # Build points list with original indices
            points = [
                (row[x], row[y], idx)
                for idx, row in group_df.iterrows()
                if pd.notna(row[x]) and pd.notna(row[y])
            ]

            if len(points) < 2:
                pareto_indices_by_color[color_val] = set(group_df.index)
                continue

            frontier_indices = _compute_pareto_frontier(
                points,
                pareto_direction["x"],
                pareto_direction["y"],
            )
            pareto_indices_by_color[color_val] = set(frontier_indices)

    # Save original plot_df for Pareto line drawing (before filtering)
    plot_df_for_pareto = plot_df

    # Filter out dominated points if requested
    if hide_pareto_dominated:
        all_pareto_indices = set()
        for indices in pareto_indices_by_color.values():
            all_pareto_indices.update(indices)

        # Also keep points with marker values in show_dominated_markers
        keep_mask = plot_df.index.isin(all_pareto_indices)
        if show_dominated_markers and marker_col is not None:
            keep_mask = keep_mask | plot_df[marker_col].isin(show_dominated_markers)

        plot_df = plot_df[keep_mask]

    # Track legend entries (only add each unique color/marker combo once)
    legend_entry_map = {}  # (color_val, marker_val) -> handle

    for idx, row in plot_df.iterrows():
        color_val = row[color_col] if color_col is not None else None
        marker_val = row[marker_col] if marker_col is not None else None

        # Skip if color/marker value is NaN
        if color_col is not None and pd.isna(color_val):
            continue
        if marker_col is not None and pd.isna(marker_val):
            continue

        color = get_color(color_col, color_val, color_index.get(color_val, 0))
        marker = (
            get_marker(marker_col, marker_val, marker_index.get(marker_val, 0))
            if marker_val is not None
            else "o"
        )

        handle = ax.scatter(
            row[x],
            row[y],
            c=[color],
            marker=marker,
            s=marker_size,
            zorder=3,
        )

        # Track for legend (only first occurrence of each combo)
        legend_key = (color_val, marker_val)
        if legend_key not in legend_entry_map:
            legend_entry_map[legend_key] = handle

    # Draw Pareto frontier step lines if pareto_line is True
    if pareto_line:
        linewidth = BASE_LINEWIDTH * scale

        for color_val in color_values:
            if color_val not in pareto_indices_by_color:
                continue

            frontier_indices = pareto_indices_by_color[color_val]
            if len(frontier_indices) < 2:
                continue

            # Get frontier points from original plot_df (before filtering)
            frontier_points = [
                (plot_df_for_pareto.loc[idx, x], plot_df_for_pareto.loc[idx, y])
                for idx in frontier_indices
                if idx in plot_df_for_pareto.index
                and pd.notna(plot_df_for_pareto.loc[idx, x])
                and pd.notna(plot_df_for_pareto.loc[idx, y])
            ]

            if len(frontier_points) < 2:
                continue

            # Sort by x (ascending for "lower is better", descending for "higher is better")
            reverse_x = pareto_direction["x"] == "higher"
            frontier_points.sort(key=lambda p: p[0], reverse=reverse_x)

            # Get line color
            line_color = get_color(color_col, color_val, color_index.get(color_val, 0))

            # Draw step function: for each consecutive pair, draw horizontal then vertical
            # For "lower x, higher y is better": go left-to-right, step down
            # For "higher x, higher y is better": go right-to-left, step down
            step_x = []
            step_y = []
            for i, (px, py) in enumerate(frontier_points):
                if i == 0:
                    step_x.append(px)
                    step_y.append(py)
                else:
                    # Add horizontal segment to new x, then vertical to new y
                    prev_y = step_y[-1]
                    step_x.append(px)
                    step_y.append(prev_y)  # Horizontal at previous y
                    step_x.append(px)
                    step_y.append(py)  # Vertical to new y

            ax.plot(
                step_x,
                step_y,
                color=line_color,
                linewidth=linewidth,
                zorder=2,  # Below points (zorder=3)
            )

    # Draw arrows between paired points if arrow_by is specified
    if arrow_by is not None:
        # Determine pairing columns: all columns except x, y, arrow_by, and metric columns
        pair_cols = [
            col for col in plot_df.columns
            if col not in {x, y, arrow_by, "instructions"}
            and not col.endswith("_reward")
            and col not in {"hacking_rate", "validation_score"}
        ]

        for _, group in plot_df.groupby(pair_cols, dropna=False):
            if len(group) != 2:
                continue
            from_row = group[group[arrow_by] == False]
            to_row = group[group[arrow_by] == True]
            if len(from_row) == 1 and len(to_row) == 1:
                x0, y0 = from_row[x].iloc[0], from_row[y].iloc[0]
                x1, y1 = to_row[x].iloc[0], to_row[y].iloc[0]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->,head_length=0.8,head_width=0.5",
                        color="red",
                        alpha=0.7,
                        lw=1.5 * scale,
                    ),
                    zorder=4,  # Above points (zorder=3)
                )

    # Auto-detect axis limits: use 0-1 for metrics, data range for other columns
    x_vals = plot_df[x].dropna()
    y_vals = plot_df[y].dropna()
    if x_vals.min() >= 0 and x_vals.max() <= 1:
        ax.set_xlim(0, 1)
    if y_vals.min() >= 0 and y_vals.max() <= 1:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Show x-axis labels and ticks only for bottommost plots
    if show_chrome and is_bottommost:
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    # Show y-axis labels and ticks only for leftmost plots
    if show_chrome and is_leftmost:
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if title:
        ax.set_title(title, fontsize=label_fontsize)

    # Build legend entries
    legend_entries = []
    if legend_entry_map:
        # Collect unique color and marker values
        color_vals = {cv for cv, _ in legend_entry_map.keys() if cv is not None}
        marker_vals = {mv for _, mv in legend_entry_map.keys() if mv is not None}

        legend_entries = _build_style_legend_entries(
            ax,
            style_specs=[
                (color_col, color_vals, color_index, get_color, {}),
                (marker_col, marker_vals, marker_index, get_marker, {"s": marker_size}),
            ],
        )

    return legend_entries


def _draw_binned_line_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    linestyle_col: str = None,
    n_bins: int = 10,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
    equal_weight_by: str = None,
    show_error_bars: bool = False,
    show_counts: bool = False,
    is_leftmost: bool = True,
    is_bottommost: bool = True,
) -> list[LegendEntry]:
    """Draw a line plot with x values binned and y values averaged within each bin.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame with data
        x: Column for x-axis values (will be binned)
        y: Column for y-axis values (will be averaged within each bin)
        hue: Column for color grouping (optional)
        linestyle_col: Column for line style grouping (optional)
        n_bins: Number of bins for x-axis
        title: Optional title for the plot
        scale: Scale factor for fonts and line widths (1.0 = full size)
        show_chrome: Whether to show axis labels
        equal_weight_by: Column to weight equally when averaging (optional).
            When set, computes per-group means first, then averages those means.
        show_error_bars: Whether to show ± SEM as error bars at each point
            (only supported when equal_weight_by is None).
        show_counts: Whether to show sample counts as text labels at each point.
        is_leftmost: Whether this plot is in the leftmost column of the grid
        is_bottommost: Whether this plot is in the bottommost row of the grid

    Returns:
        List of LegendEntry for legend
    """
    legend_entries = []

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return legend_entries

    # Scale font and line sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    linewidth = BASE_LINEWIDTH * scale
    marker_size = BASE_MARKER_SIZE * scale

    # Create bin edges and labels
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(n_bins)]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Add bin column to dataframe
    df = df.copy()
    df["_bin"] = pd.cut(
        df[x],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
    )

    # Handle optional hue and linestyle
    hue_values = sort_values(hue, list(df[hue].dropna().unique())) if hue is not None else [None]
    linestyle_values = (
        sort_values(linestyle_col, list(df[linestyle_col].dropna().unique()))
        if linestyle_col is not None
        else [None]
    )

    hue_index = {val: i for i, val in enumerate(hue_values)}
    linestyle_index = {val: j for j, val in enumerate(linestyle_values)}

    # Track which values were actually drawn
    drawn_hue_vals = set()
    drawn_linestyle_vals = set()

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val] if hue_val is not None else df

        if hue_df.empty:
            continue

        for j, linestyle_val in enumerate(linestyle_values):
            # Filter by linestyle column if applicable
            series_df = (
                hue_df[hue_df[linestyle_col] == linestyle_val]
                if linestyle_val is not None
                else hue_df
            )

            if series_df.empty:
                continue

            # Filter to rows that have valid y values
            series_df = series_df.dropna(subset=[y])
            if series_df.empty:
                continue

            # Group by bin and compute mean (and std/count for error bands)
            if equal_weight_by is not None:
                # Macro average: first compute per-group means, then average those
                # Missing groups within a bin are treated as 0
                per_group = series_df.groupby(["_bin", equal_weight_by], observed=True)[y].mean()

                # Reindex to include all (bin, group) combinations, filling missing with 0
                all_bins = series_df["_bin"].cat.categories
                all_groups = series_df[equal_weight_by].unique()
                full_index = pd.MultiIndex.from_product([all_bins, all_groups], names=["_bin", equal_weight_by])
                per_group = per_group.reindex(full_index, fill_value=0)

                binned = per_group.groupby("_bin", observed=True).agg(["mean", "count"])
            else:
                # Micro average: pool all data and compute mean directly
                binned = series_df.groupby("_bin", observed=True)[y].agg(["mean", "std", "count"])

            # Get color and linestyle
            color = get_color(hue, hue_val, i)
            ls = (
                get_linestyle(linestyle_col, linestyle_val, j)
                if linestyle_val is not None
                else "-"
            )

            # Build x,y coordinates (skip empty bins)
            x_vals = []
            y_vals = []
            sem_vals = []  # Standard error of mean (None if can't be calculated)
            count_vals = []  # Sample counts
            for k, bin_label in enumerate(bin_labels):
                if bin_label in binned.index and binned.loc[bin_label, "count"] > 0:
                    x_vals.append(bin_centers[k])
                    y_vals.append(binned.loc[bin_label, "mean"])
                    count_vals.append(int(binned.loc[bin_label, "count"]))
                    if show_error_bars and equal_weight_by is None and "std" in binned.columns:
                        count = binned.loc[bin_label, "count"]
                        std = binned.loc[bin_label, "std"]
                        # Only compute SEM if we have enough samples
                        if count > 1 and pd.notna(std):
                            sem_vals.append(std / np.sqrt(count))
                        else:
                            sem_vals.append(None)

            if x_vals:
                # Draw error bars first (so line is on top)
                if show_error_bars and equal_weight_by is None and sem_vals:
                    for px, py, sem in zip(x_vals, y_vals, sem_vals):
                        if sem is not None:
                            ax.errorbar(
                                px, py, yerr=sem,
                                color=color, alpha=0.5,
                                capsize=3, capthick=1, linewidth=1,
                                fmt="none",  # Don't draw marker, just error bar
                            )

                (line,) = ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    linestyle=ls,
                    linewidth=linewidth,
                    marker="o",
                    markersize=np.sqrt(marker_size),
                )

                # Draw count labels
                if show_counts:
                    count_fontsize = tick_fontsize * 0.7
                    for px, py, count in zip(x_vals, y_vals, count_vals):
                        ax.annotate(
                            str(count),
                            (px, py),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            fontsize=count_fontsize,
                            color=color,
                        )

                if hue_val is not None:
                    drawn_hue_vals.add(hue_val)
                if linestyle_val is not None:
                    drawn_linestyle_vals.add(linestyle_val)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(bin_edges)
    ax.grid(True, alpha=0.3)

    # Show x-axis labels and ticks only for bottommost plots
    if show_chrome and is_bottommost:
        ax.set_xticklabels([f"{e:.1f}" for e in bin_edges])
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
    else:
        ax.set_xticklabels([])

    # Show y-axis labels and ticks only for leftmost plots
    if show_chrome and is_leftmost:
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if title:
        ax.set_title(title, fontsize=label_fontsize)

    # Build legend entries with separate sections for color and linestyle
    legend_entries = _build_style_legend_entries(
        ax,
        style_specs=[
            (hue, drawn_hue_vals, hue_index, get_color, {}),
            (linestyle_col, drawn_linestyle_vals, linestyle_index, get_linestyle, {"linewidth": linewidth}),
        ],
    )

    return legend_entries


def _build_step_function(
    x_vals: np.ndarray, y_vals: np.ndarray, end_x: float
) -> tuple[list, list]:
    """Build step function coordinates from discrete points.

    Args:
        x_vals: Array of x values
        y_vals: Array of y values
        end_x: An x value to extend the line to (for run endpoints)
    """
    if len(x_vals) == 0:
        return [], []

    result_x = []
    result_y = []

    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        if i > 0:
            # Horizontal line from previous x to current x
            result_x.append(x)
            result_y.append(result_y[-1])

        result_x.append(x)
        result_y.append(y)

    # Extend line to end_x if provided and greater than last x
    if result_x and end_x > result_x[-1]:
        result_x.append(end_x)
        result_y.append(result_y[-1])

    return result_x, result_y


def _average_step_functions(lines: list[tuple[list, list]]) -> tuple[list, list]:
    """Average multiple step functions."""
    if not lines:
        return [], []

    # Collect all x change points
    all_x_points = set()
    for x_vals, _ in lines:
        all_x_points.update(x_vals)
    all_x_points = sorted(all_x_points)

    avg_x = []
    avg_y = []

    for x in all_x_points:
        y_sum = 0
        y_count = 0

        for x_vals, y_vals in lines:
            # Find the y value at this x (step function lookup)
            for j in range(len(x_vals)):
                if x_vals[j] <= x:
                    if j == len(x_vals) - 1 or x < x_vals[j + 1]:
                        y_sum += y_vals[j]
                        y_count += 1
                        break

        if y_count > 0:
            avg = y_sum / y_count

            # Add horizontal segment if needed
            if avg_y and avg_y[-1] != avg:
                avg_x.append(x)
                avg_y.append(avg_y[-1])

            avg_x.append(x)
            avg_y.append(avg)

    return avg_x, avg_y


# =============================================================================
# Config Classes
# =============================================================================


@dataclass
class Figure:
    """Configuration for a single figure."""

    layout: LayoutNode
    name: str | None = None
    filter: str | Callable[[pd.DataFrame], pd.Series] = None
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    legend_ncol: int | None = None

    def render(
        self, dfs: DFs, figure_index: int = 1
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """Render this figure.

        Args:
            dfs: Dict with "main" (candidate rows) and "gepa_runs" (run-level data)
            figure_index: 1-based index of this figure (used for warning messages)

        Returns:
            Tuple of (figure, filtered_df) where filtered_df is the main DataFrame after filtering
        """
        filtered_dfs = _filter_main(dfs, self.filter)
        if self.transform is not None:
            filtered_dfs = {**filtered_dfs, "main": self.transform(filtered_dfs["main"])}

        # Get grid size info
        grid_size = self.layout.get_grid_size(filtered_dfs["main"])

        # Create figure with appropriate size
        fig_width = grid_size.cols * grid_size.cell_width
        fig_height = grid_size.rows * grid_size.cell_height
        fig = plt.figure(figsize=(fig_width, fig_height), layout='constrained')

        if self.name:
            fig.suptitle(self.name, fontsize=14)

        gs = fig.add_gridspec(grid_size.rows, grid_size.cols)

        # Render layout with scale=1.0 at the top level
        result = self.layout.render(
            fig,
            gs,
            slice(0, grid_size.rows),
            slice(0, grid_size.cols),
            filtered_dfs,
            scale=1.0,
            subtitle=None,
        )

        # Print any aggregation warnings
        _print_aggregation_warnings(result.warnings, figure_index)

        # Add figure-level legend outside the plots if there are legend entries
        if result.legend_entries:
            handles = [entry.handle for entry in result.legend_entries]
            labels = [entry.label for entry in result.legend_entries]
            ncol = self.legend_ncol if self.legend_ncol is not None else min(len(labels), 4)
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=ncol,
                fontsize=BASE_LEGEND_FONTSIZE,
            )

        return fig, filtered_dfs["main"]


@dataclass
class PlotConfig:
    """Configuration for loading and plotting experiment data.

    Attributes:
        paths: List of experiment paths to load data from. Supports both GEPA-optimized
            experiments and baseline-only experiments (prompter_name=None).
        figures: List of Figure configurations to render.
        quick_mode: Controls which candidates are evaluated during data loading.
            If True, only evaluates the first and last candidates (faster, suitable for bar plots).
            If False, evaluates all improving candidates (needed for progression plots).
        computed_columns: Dict mapping column names to functions that compute them from the DataFrame.
        metrics: List of metric names to compute. If None, computes all metrics.
            Use this to speed up loading when you only need specific metrics.

    Example:
        config = PlotConfig(
            paths=["logs/mcq/2025-.../"],
            figures=[Figure(layout=BarPlot(x="env_name", y="proxy_reward"))],
        )
    """

    paths: list[str]
    figures: list[Figure]
    quick_mode: bool = False
    computed_columns: dict[str, Callable[[pd.DataFrame], pd.Series]] = field(
        default_factory=dict
    )
    metrics: list[str] | None = None

    _dfs: DFs = field(default=None, init=False, repr=False)

    def load_dfs(self) -> DFs:
        """Load and cache the DataFrames.

        Returns:
            Dict with "main" DataFrame (and "gepa_runs" for GEPA data)
        """
        if self._dfs is None:
            self._dfs = load_dfs(
                self.paths, quick_mode=self.quick_mode, metrics=self.metrics
            )

            # Apply computed columns to main DataFrame only
            for col_name, col_fn in self.computed_columns.items():
                self._dfs["main"][col_name] = col_fn(self._dfs["main"])

        return self._dfs

    def render_all(self) -> list[pd.DataFrame]:
        """Render all figures.

        Returns:
            List of filtered DataFrames, one per figure in order.
        """
        dfs = self.load_dfs()
        filtered_dfs = []

        for idx, figure in enumerate(self.figures, start=1):
            fig, filtered_df = figure.render(dfs, figure_index=idx)
            filtered_dfs.append(filtered_df)
            plt.show()

        return filtered_dfs
