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
import numpy as np
import pandas as pd

from core.experiment.df import load_experiment_dfs, DFs
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
    # Runtime/internal fields that don't affect experiment outcomes:
    "seed",
    "log_dir_index",
    "cache",
    "num_threads",
} | set(METRICS.keys())


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
        # Skip expected varying columns and explicitly grouped columns
        if col in EXPECTED_VARYING_COLUMNS or col in grouped_columns:
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


def _print_aggregation_warnings(warnings: dict[str, list], figure_name: str) -> None:
    """Print aggregation warnings if any exist."""
    if not warnings:
        return

    print(
        f"\n⚠️  WARNING: Figure '{figure_name}' is aggregating over multiple values of:"
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

# Maps column names to friendly display names for axis labels, titles, etc.
COLUMN_DISPLAY_NAMES: dict[str, str] = {
    "discovery_eval_counts": "Task LM Calls",
    "reflection_call_count": "Reflection LM Calls",
    "column_name": "Column",
    "column_value": "Value",
    "suggest_hack": "Told to Hack?",
    "is_sanitized": "Sanitized?",
    "hint_type": "Hint Type",
    "subset": "Subset",
    "run_index": "Run",
    "candidate_index": "Candidate",
    "prompt_type": "Prompt Type",  # Doesn't exist by default
    "proxy_reward": "Proxy Reward",
    "true_reward": "True Reward",
    "hacking_rate": "Hacking Rate",
    "prompt_verbalizes": "Verbalizes Hacking?",
    "prompt_verbalizes_float": "Hack Verbalization Level",
    "cot_verbalizes_yes": "CoT Verbalizes Hacking (Yes)",
    "cot_verbalizes_unclear": "CoT Verbalizes Hacking (Unclear)",
    "cot_verbalizes_no": "CoT Verbalizes Hacking (No)",
    "prompt_suspiciousness": "Prompt Suspicious?",
    "cot_suspiciousness_yes": "CoT Suspicious (Yes)",
    "cot_suspiciousness_unclear": "CoT Suspicious (Unclear)",
    "cot_suspiciousness_no": "CoT Suspicious (No)",
}

# Maps (column, value) pairs to friendly display names for legends, tick labels, etc.
VALUE_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "column_name": {
        "proxy_reward": "Proxy Reward",
        "true_reward": "True Reward",
        "hacking_rate": "Hacking Rate",
        "prompt_verbalizes": "Verbalizes Hacking?",
    },
    "is_sanitized": {
        True: "Sanitized",
        False: "Original",
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
}


def get_display_name(column: str) -> str:
    """Get user-friendly display name for a column.

    Args:
        column: Internal column name

    Returns:
        User-friendly display name, or the original column name if no mapping exists.
    """
    return COLUMN_DISPLAY_NAMES.get(column, column)


def get_display_value(column: str, value: Any) -> str:
    """Get user-friendly display name for a column value.

    Args:
        column: Column the value belongs to
        value: Internal value

    Returns:
        User-friendly display name, or str(value) if no mapping exists.
    """
    column_mappings = VALUE_DISPLAY_NAMES.get(column, {})
    return column_mappings.get(value, str(value))


# =============================================================================
# Value Ordering
# =============================================================================

# Defines the canonical order for categorical values.
# Values not in this list will be sorted alphabetically after the listed ones.
VALUE_ORDER: dict[str, list] = {
    "prompt_type": [
        "baseline",
        "no_hack",
        "no_hack_sanitized",
        "explicit_hack",
        "explicit_hack_sanitized",
    ],
    "column_name": ["proxy_reward", "true_reward", "prompt_verbalizes", "hacking_rate"],
    "prompt_verbalizes": ["no", "unclear", "yes"],
    "optimizer": [
        "Baseline",
        "GEPA",
        "GEPA (Hack)",
        "GEPA (Hack + Teacher)",
        "GEPA (Hack + Teacher + Tool)",
        "RL",
    ],
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
        return sorted(values)

    order = VALUE_ORDER[column]
    order_map = {v: i for i, v in enumerate(order)}

    def sort_key(v):
        if v in order_map:
            return (0, order_map[v])
        return (1, str(v))  # Unlisted values go at end, sorted alphabetically

    return sorted(values, key=sort_key)


# =============================================================================
# Style Registry
# =============================================================================

# Maps (column, value) pairs to style attributes that should always be used.
# When plotting, if a value has a pinned style, it overrides automatic assignment.
# Supported style keys: "color", "linestyle", "marker"
STYLE_REGISTRY: dict[tuple[str, Any], dict[str, Any]] = {
    # Colors for column names (used when melting columns)
    ("column_name", "proxy_reward"): {"color": "#ff7777"},
    ("column_name", "true_reward"): {"color": "#66b3ff"},
    ("column_name", "prompt_verbalizes"): {"color": "#77dd77"},
    ("column_name", "hacking_rate"): {"color": "#ffaa44"},
    # Colors for prompt_verbalizes values (red=no, yellow=unclear, green=yes)
    ("prompt_verbalizes", "no"): {"color": "#ff4444"},
    ("prompt_verbalizes", "unclear"): {"color": "#ffcc00"},
    ("prompt_verbalizes", "yes"): {"color": "#44bb44"},
    # Line styles for is_sanitized
    ("is_sanitized", False): {"linestyle": "-"},
    ("is_sanitized", True): {"linestyle": ":"},
    # Colors for optimizer values (used in bar/progression plots)
    ("optimizer", "Baseline"): {"color": "#7f7f7f"},  # gray
    ("optimizer", "GEPA"): {"color": "#1f77b4"},  # blue
    ("optimizer", "GEPA (Hack)"): {"color": "#ff7f0e"},  # orange
    ("optimizer", "GEPA (Hack + Teacher)"): {"color": "#2ca02c"},  # green
    ("optimizer", "GEPA (Hack + Teacher + Tool)"): {"color": "#d62728"},  # red
    ("optimizer", "RL"): {"color": "#9467bd"},  # purple
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


def _build_label(
    col1: str, val1: Any, col2: str = None, val2: Any = None
) -> str | None:
    """Build a legend label from one or two (column, value) pairs.

    Args:
        col1: First column name (or None)
        val1: First value (or None)
        col2: Second column name (optional)
        val2: Second value (optional)

    Returns:
        Label string like "Value1 (Value2)" or just "Value1", or None if both are None
    """
    display1 = get_display_value(col1, val1) if val1 is not None else None
    display2 = get_display_value(col2, val2) if val2 is not None else None

    if display1 and display2:
        return f"{display1} ({display2})"
    return display1 or display2


# =============================================================================
# Layout Nodes (composable plotting components)
# =============================================================================


class GridSize(NamedTuple):
    """Size information returned by LayoutNode.get_grid_size().

    Attributes:
        rows: Total rows in grid units
        cols: Total columns in grid units
        ref_rows: Rows occupied by the "main" or reference plot
        ref_cols: Columns occupied by the "main" or reference plot
    """

    rows: int
    cols: int
    ref_rows: int
    ref_cols: int


class LayoutNode(ABC):
    """Base class for composable layout nodes."""

    @abstractmethod
    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        """Return grid size information for this layout.

        Returns:
            GridSize with total dimensions and reference plot dimensions.
            The reference dimensions indicate how many cells a "full-sized" plot occupies,
            which is used to compute cell size so that main plots are always REFERENCE_WIDTH x REFERENCE_HEIGHT.
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
    ) -> dict[str, list]:
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

        Returns:
            Dict of aggregation warnings (column -> unique values) from all leaf plots.
        """
        pass


@dataclass
class Grid(LayoutNode):
    """Creates a grid of plots, one per unique value of groupby column."""

    groupby: str
    cols_wrap: int = 2
    inner: LayoutNode = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        unique_values = df[self.groupby].dropna().unique()
        n_groups = len(unique_values)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)
        grid_rows = ceil(n_groups / self.cols_wrap)

        return GridSize(
            rows=grid_rows * inner_size.rows,
            cols=grid_cols * inner_size.cols,
            ref_rows=inner_size.ref_rows,  # Pass through from inner
            ref_cols=inner_size.ref_cols,
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
    ) -> dict[str, list]:
        df = dfs["main"]
        unique_values = sorted(df[self.groupby].dropna().unique())
        n_groups = len(unique_values)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)

        all_warnings = defaultdict(set)

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
            # Format: "column = value" and chain with parent subtitle if present
            display_value = get_display_value(self.groupby, value)
            this_level = f"{self.groupby} = {display_value}"
            cell_subtitle = f"{subtitle}, {this_level}" if subtitle else this_level

            if self.inner is None:
                # No inner layout - just create a placeholder
                ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
                ax.set_title(cell_subtitle)
                ax.text(0.5, 0.5, "No inner layout", ha="center", va="center")
            else:
                # Grid passes scale and show_chrome through unchanged
                warnings = self.inner.render(
                    fig,
                    gs,
                    slice(r_start, r_end),
                    slice(c_start, c_end),
                    sub_dfs,
                    scale,
                    show_chrome,
                    subtitle=cell_subtitle,
                )
                for col, vals in warnings.items():
                    all_warnings[col].update(vals)

        # Convert sets back to lists
        return {col: list(vals) for col, vals in all_warnings.items()}


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
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        # Main plot: n_sides cols × n_sides rows
        # Side plots: 1 col × 1 row each, stacked vertically
        # This gives uniform 1/n_sides scaling in both dimensions for sides
        total_rows = n_sides * inner_size.rows
        total_cols = (n_sides + 1) * inner_size.cols  # n_sides for main, 1 for sides

        # Reference size is the main plot size (n_sides × n_sides cells)
        ref_rows = n_sides * inner_size.ref_rows
        ref_cols = n_sides * inner_size.ref_cols

        return GridSize(total_rows, total_cols, ref_rows, ref_cols)

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
    ) -> dict[str, list]:
        df = dfs["main"]
        unique_values = list(df[self.groupby].unique())

        # Separate main and side values
        if self.main_value is None:
            # main_value=None means use rows where groupby column is NaN
            main_dfs = _filter_main(dfs, lambda df: df[self.groupby].isna())
            side_values = sorted([v for v in unique_values if pd.notna(v)])
        else:
            main_dfs = _filter_main(dfs, lambda df: df[self.groupby] == self.main_value)
            side_values = sorted(
                [v for v in unique_values if v != self.main_value and pd.notna(v)]
            )

        n_sides = max(len(side_values), 1)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        all_warnings = defaultdict(set)

        # Main plot region: left n_sides/(n_sides+1), full height
        main_col_end = col_slice.start + n_sides * inner_size.cols

        if self.inner is None:
            ax_main = fig.add_subplot(gs[row_slice, col_slice.start : main_col_end])
            ax_main.set_title("Main (no inner layout)")
        else:
            # Main plot gets the current scale and show_chrome unchanged
            # Pass through subtitle from parent
            warnings = self.inner.render(
                fig,
                gs,
                row_slice,
                slice(col_slice.start, main_col_end),
                main_dfs,
                scale,
                show_chrome,
                subtitle=subtitle,
            )
            for col, vals in warnings.items():
                all_warnings[col].update(vals)

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
                warnings = self.inner.render(
                    fig,
                    gs,
                    slice(r_start, r_end),
                    slice(main_col_end, col_slice.stop),
                    side_dfs,
                    side_scale,
                    show_chrome=False,
                )
                for col, vals in warnings.items():
                    all_warnings[col].update(vals)

        # Convert sets back to lists
        return {col: list(vals) for col, vals in all_warnings.items()}


# Reference size for a "full-sized" plot in inches
# A main plot (or standalone leaf plot) will always be this size
REFERENCE_WIDTH = 12
REFERENCE_HEIGHT = 6


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
    hue: str = None
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        # Leaf plot: 1x1 cell, and it IS the reference size
        return GridSize(1, 1, 1, 1)

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
    ) -> dict[str, list]:
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
        _draw_bar_plot(ax, df, self.x, y_col, self.hue, title, scale, show_chrome)
        return _check_aggregation_warnings(df, self.x, y_col, self.hue, type(self))


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
        title: Optional title for the plot
        show_individual_lines: Whether to show faint lines for individual runs

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
    """

    x: str = "discovery_eval_counts"
    y: str | list[str] = None  # Required, but default None for backwards compat error
    hue: str = None
    linestyle: str = None
    title: str = None
    show_individual_lines: bool = True

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        # Leaf plot: 1x1 cell, and it IS the reference size
        return GridSize(1, 1, 1, 1)

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
    ) -> dict[str, list]:
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
        _draw_progression_plot(
            ax,
            plot_dfs,
            self.x,
            y_col,
            self.hue,
            self.linestyle,
            title,
            scale,
            show_chrome,
            self.show_individual_lines,
        )
        return _check_aggregation_warnings(
            df, self.x, y_col, self.hue, type(self), self.linestyle
        )


@dataclass
class ScatterPlot(LayoutNode):
    """Scatter plot comparing two metrics with optional color and marker grouping.

    Args:
        x: Column for x-axis values
        y: Column for y-axis values
        color: Column for color grouping (optional)
        marker: Column for marker shape grouping (optional)
        aggregate_by: Columns to aggregate by (optional). If None, shows individual
            points (one per row). If specified, shows one point per unique combination
            of the aggregate_by columns, with x/y values being the mean.
            Must include color and marker columns if those are specified.
        title: Optional title for the plot

    Examples:
        # Individual points, colored by prompt_verbalizes
        ScatterPlot(x="proxy_reward", y="true_reward", color="prompt_verbalizes")

        # Aggregated: one point per prompt_verbalizes value
        ScatterPlot(x="proxy_reward", y="true_reward", color="prompt_verbalizes",
                    aggregate_by=["prompt_verbalizes"])

        # Aggregated by multiple columns
        ScatterPlot(x="proxy_reward", y="true_reward",
                    color="prompt_verbalizes", marker="suggest_hack",
                    aggregate_by=["prompt_verbalizes", "suggest_hack"])
    """

    x: str
    y: str
    color: str = None
    marker: str = None
    aggregate_by: list[str] = None
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        # Leaf plot: 1x1 cell, and it IS the reference size
        return GridSize(1, 1, 1, 1)

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
    ) -> dict[str, list]:
        df = dfs["main"]

        # Validate aggregate_by includes color/marker if specified
        if self.aggregate_by is not None:
            if self.color is not None and self.color not in self.aggregate_by:
                raise ValueError(
                    f"aggregate_by must include color column '{self.color}'"
                )
            if self.marker is not None and self.marker not in self.aggregate_by:
                raise ValueError(
                    f"aggregate_by must include marker column '{self.marker}'"
                )

        # Use subtitle from parent Grid if no explicit title set
        title = self.title if self.title is not None else subtitle

        ax = fig.add_subplot(gs[row_slice, col_slice])
        _draw_scatter_plot(
            ax,
            df,
            self.x,
            self.y,
            self.color,
            self.marker,
            self.aggregate_by,
            title,
            scale,
            show_chrome,
        )
        return _check_aggregation_warnings(
            df, self.x, self.y, self.color, type(self), self.marker
        )


# =============================================================================
# Layout Helpers
# =============================================================================


def with_subsets(plot: LayoutNode) -> MainSide:
    """Wrap a plot with main/side panels for subsets."""
    return MainSide(groupby="subset", main_value=None, inner=plot)


def full_layout(plot: LayoutNode) -> Grid:
    """Standard layout: grid by hint_type, main/side by subset."""
    return Grid(groupby="hint_type", cols_wrap=2, inner=with_subsets(plot))


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


def make_verbalization_float_column(df: pd.DataFrame) -> pd.Series:
    """Convert prompt_verbalizes string values to floats for plotting.

    Maps:
        "yes" -> 1.0
        "unclear" -> 0.5
        "no" -> 0.0

    Works with wide-format DataFrames where prompt_verbalizes is a column.
    Returns the converted values; use this to overwrite the prompt_verbalizes column:
        df["prompt_verbalizes"] = make_verbalization_float_column(df)
    """
    verbalization_map = {"yes": 1.0, "unclear": 0.5, "no": 0.0}

    if "prompt_verbalizes" not in df.columns:
        raise ValueError("DataFrame must have 'prompt_verbalizes' column")

    return df["prompt_verbalizes"].map(
        lambda v: verbalization_map.get(v, v) if pd.notna(v) else v
    )


def make_optimizer_column(df: pd.DataFrame) -> pd.Series:
    """Create an 'optimizer' column distinguishing Baseline, GEPA variants, and RL.

    Categories:
        - "Baseline" - First candidate (is_baseline=True)
        - "RL" - Baseline-only experiment with tinker model (prompter_name=None, executor starts with tinker/)
        - "GEPA" - GEPA-optimized with suggest_hack="no"
        - "GEPA (Hack)" - GEPA-optimized with suggest_hack="explicit", no teacher
        - "GEPA (Hack + Teacher)" - suggest_hack="explicit" with train_teacher_file
        - "GEPA (Hack + Teacher + Tool)" - suggest_hack="explicit" with train_teacher_file and teacher_tool_model

    This is useful for bar plots comparing different optimization approaches.

    Usage:
        PlotConfig(
            paths=["logs/mcq/..."],
            computed_columns={"optimizer": make_optimizer_column},
            figures=[
                Figure(
                    name="Comparison",
                    filter=lambda df: df.optimizer.notna(),
                    layout=BarPlot(x="hint_type", y="proxy_reward", hue="optimizer"),
                ),
            ],
        )
    """

    def classify_row(row):
        # Check if this is a baseline-only experiment (no prompter)
        if row.get("prompter_name") is None:
            executor = row.get("executor_name", "")
            if executor and executor.startswith("tinker/"):
                return "RL"
            raise ValueError(
                f"Cannot classify baseline-only experiment with non-tinker executor: {executor}."
            )

        # Check if this is a baseline candidate within a GEPA run
        if row.get("is_baseline"):
            return "Baseline"

        # GEPA variants based on config
        suggest_hack = row.get("suggest_hack")
        has_teacher = row.get("train_teacher_file") is not None
        has_tool = row.get("teacher_tool_model") is not None

        if suggest_hack == "explicit":
            if has_teacher and has_tool:
                return "GEPA (Hack + Teacher + Tool)"
            elif has_teacher:
                return "GEPA (Hack + Teacher)"
            return "GEPA (Hack)"
        elif suggest_hack == "no":
            return "GEPA"

        raise ValueError(
            f"Cannot classify GEPA experiment with suggest_hack={suggest_hack!r}. "
        )

    return df.apply(classify_row, axis=1)


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
) -> None:
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
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Get unique x values, sorted by canonical order
    x_values = sort_values(x, list(df[x].unique()))

    # Handle optional hue
    hue_values = sort_values(hue, list(df[hue].unique())) if hue is not None else [None]

    n_x = len(x_values)
    n_hue = len(hue_values)

    if n_x == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Scale font and marker sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    legend_fontsize = BASE_LEGEND_FONTSIZE * scale
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

        # Add bar labels
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=bar_label_fontsize)

        # Overlay individual data points
        for j, (pos, individuals) in enumerate(zip(bar_positions, all_individuals)):
            if len(individuals) > 0:
                ax.scatter(
                    np.full(len(individuals), pos),
                    individuals,
                    marker="x",
                    color="#202020",
                    s=marker_size,
                    linewidth=0.5,
                    zorder=3,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [get_display_value(x, v) for v in x_values], fontsize=tick_fontsize
    )
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    if show_chrome:
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)
        if hue is not None:
            ax.legend(fontsize=legend_fontsize)

    if title:
        ax.set_title(title, fontsize=label_fontsize)


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
) -> None:
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
        ax.plot(
            avg_x,
            avg_y,
            color=color,
            alpha=1.0,
            linewidth=linewidth,
            linestyle=ls,
            label=label,
        )


def _draw_progression_plot(
    ax: plt.Axes,
    dfs: DFs,
    x: str,
    y: str,
    hue: str,
    linestyle_col: str = None,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
    show_individual_lines: bool = True,
) -> None:
    """Draw a progression plot with step functions.

    Args:
        ax: Matplotlib axes to draw on
        dfs: Dict with "main" (candidate rows) and "gepa_runs" (run-level data)
        x: Column for x-axis
        y: Column for y-axis values
        hue: Column for color grouping (can be None)
        linestyle_col: Column for line style grouping (optional)
        title: Optional title for the plot
        scale: Scale factor for fonts and line widths (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
        show_individual_lines: Whether to show faint lines for individual run_index values
    """
    df = dfs["main"]
    gepa_runs = dfs["gepa_runs"]

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

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
    legend_fontsize = BASE_LEGEND_FONTSIZE * scale
    linewidth = BASE_LINEWIDTH * scale
    faint_linewidth = BASE_FAINT_LINEWIDTH * scale

    # Handle optional hue and linestyle
    hue_values = sort_values(hue, list(df[hue].unique())) if hue is not None else [None]
    linestyle_values = (
        sort_values(linestyle_col, list(df[linestyle_col].dropna().unique()))
        if linestyle_col is not None
        else [None]
    )

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val] if hue_val is not None else df
        color = get_color(hue, hue_val, i)

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
            label = _build_label(hue, hue_val, linestyle_col, linestyle_val)

            _draw_progression_series(
                ax,
                series_df,
                x,
                y,
                run_end_xs,
                color,
                ls,
                label,
                linewidth,
                faint_linewidth,
                show_individual_lines,
            )

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if show_chrome:
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)
        if hue is not None or linestyle_col is not None:
            ax.legend(loc="best", fontsize=legend_fontsize)

    if title:
        ax.set_title(title, fontsize=label_fontsize)


def _draw_scatter_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str = None,
    marker_col: str = None,
    aggregate_by: list[str] = None,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
) -> None:
    """Draw a scatter plot with optional aggregation.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame with data
        x: Column for x-axis values
        y: Column for y-axis values
        color_col: Column for color grouping (optional)
        marker_col: Column for marker shape grouping (optional)
        aggregate_by: Columns to aggregate by (optional). If None, shows individual
            points. If specified, shows one point per unique combination with mean x/y.
        title: Optional title for the plot
        scale: Scale factor for fonts and markers (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Scale font and marker sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    legend_fontsize = BASE_LEGEND_FONTSIZE * scale
    marker_size = BASE_MARKER_SIZE * 3 * scale  # Larger markers for scatter

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

    # Prepare data: aggregate if requested, otherwise use as-is
    plot_df = (
        df.groupby(aggregate_by, dropna=False).agg({x: "mean", y: "mean"}).reset_index()
        if aggregate_by is not None
        else df
    )

    # Track legend entries (only add each unique color/marker combo once)
    legend_entries = {}  # (color_val, marker_val) -> handle

    for _, row in plot_df.iterrows():
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
        if legend_key not in legend_entries:
            legend_entries[legend_key] = handle

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if show_chrome:
        ax.set_xlabel(get_display_name(x), fontsize=label_fontsize)
        ax.set_ylabel(get_display_name(y), fontsize=label_fontsize)

        # Build legend from tracked entries
        if legend_entries:
            legend_handles = []
            legend_labels = []
            for (color_val, marker_val), handle in legend_entries.items():
                label = _build_label(color_col, color_val, marker_col, marker_val)
                if label:
                    legend_handles.append(handle)
                    legend_labels.append(label)

            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    fontsize=legend_fontsize,
                )

    if title:
        ax.set_title(title, fontsize=label_fontsize)


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


def _format_paths(paths: list[str]) -> str:
    """Format paths for display in figure title.

    Args:
        paths: List of experiment paths

    Returns:
        Formatted string like "logs/mcq/..." or "logs/mcq/... (+2 more)"
    """
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    return f"{paths[0]} (+{len(paths) - 1} more)"


@dataclass
class Figure:
    """Configuration for a single figure."""

    name: str
    layout: LayoutNode
    filter: str | Callable[[pd.DataFrame], pd.Series] = None

    def render(
        self, dfs: DFs, paths: list[str] = None
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """Render this figure.

        Args:
            dfs: Dict with "main" (candidate rows) and "gepa_runs" (run-level data)
            paths: Optional list of paths to display in title

        Returns:
            Tuple of (figure, filtered_df) where filtered_df is the main DataFrame after filtering
        """
        filtered_dfs = _filter_main(dfs, self.filter)

        # Get grid size info
        grid_size = self.layout.get_grid_size(filtered_dfs["main"])

        # Compute cell size so that reference plot is REFERENCE_WIDTH x REFERENCE_HEIGHT
        cell_width = REFERENCE_WIDTH / grid_size.ref_cols
        cell_height = REFERENCE_HEIGHT / grid_size.ref_rows

        # Create figure with appropriate size
        fig_width = grid_size.cols * cell_width
        fig_height = grid_size.rows * cell_height
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Build title with optional path info
        if paths:
            title = f"{self.name}\n{_format_paths(paths)}"
        else:
            title = self.name
        fig.suptitle(title, fontsize=14)

        gs = fig.add_gridspec(grid_size.rows, grid_size.cols, hspace=0.4, wspace=0.4)

        # Render layout with scale=1.0 at the top level
        warnings = self.layout.render(
            fig,
            gs,
            slice(0, grid_size.rows),
            slice(0, grid_size.cols),
            filtered_dfs,
            scale=1.0,
            subtitle=None,
        )

        # Print any aggregation warnings
        _print_aggregation_warnings(warnings, self.name)

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
        show_paths: Whether to show experiment paths in figure titles.
        metrics: List of metric names to compute. If None, computes all metrics.
            Use this to speed up loading when you only need specific metrics.

    Example comparing GEPA and baseline-only experiments:
        config = PlotConfig(
            paths=[
                "logs/mcq/2025-.../",  # GEPA-optimized
                "logs/mcq/2025-.../baseline-only/",  # Baseline-only (prompter_name=None)
            ],
            computed_columns={"optimizer": make_optimizer_column},
            ...
        )
    """

    paths: list[str]
    figures: list[Figure]
    quick_mode: bool = False
    computed_columns: dict[str, Callable[[pd.DataFrame], pd.Series]] = field(
        default_factory=dict
    )
    show_paths: bool = True
    metrics: list[str] | None = None

    _dfs: DFs = field(default=None, init=False, repr=False)

    def load_dfs(self) -> DFs:
        """Load and cache the DataFrames.

        Returns:
            Dict with "main" (candidate rows) and "gepa_runs" (run-level data)
        """
        if self._dfs is None:
            self._dfs = load_experiment_dfs(
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
        paths = self.paths if self.show_paths else None

        for figure in self.figures:
            fig, filtered_df = figure.render(dfs, paths=paths)
            filtered_dfs.append(filtered_df)
            plt.show()

        return filtered_dfs

    def render(self, name: str) -> pd.DataFrame:
        """Render a specific figure by name.

        Returns:
            The filtered DataFrame used for the figure.
        """
        dfs = self.load_dfs()
        paths = self.paths if self.show_paths else None

        for figure in self.figures:
            if figure.name == name:
                fig, filtered_df = figure.render(dfs, paths=paths)
                plt.show()
                return filtered_df

        raise ValueError(
            f"Figure '{name}' not found. Available: {[f.name for f in self.figures]}"
        )
